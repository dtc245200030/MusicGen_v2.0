import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from encodec import EncodecModel


class TextConditioner(nn.Module):
    """
    Trích xuất Text Embedding từ T5 và chiếu (Project) về kích thước d_model của Transformer.
    """

    def __init__(self, t5_model_name="t5-base", d_model=768, freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        self.t5 = T5EncoderModel.from_pretrained(t5_model_name)

        if freeze:
            self.t5.eval()
            for param in self.t5.parameters():
                param.requires_grad = False

        t5_dim = self.t5.config.d_model
        self.projector = nn.Linear(t5_dim, d_model)

    def forward(self, text_list, device):
        inputs = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = self.t5(**inputs)

        text_embeds = self.projector(outputs.last_hidden_state)
        return text_embeds, inputs.attention_mask


class MusicTransformerLM(nn.Module):
    """
    Transformer Decoder chính tự sinh Audio Tokens từ Text Condition.
    Tích hợp Classifier-Free Guidance (CFG) và EnCodec Audio Decoding.
    """

    def __init__(
            self,
            num_codebooks=4,  # Số lượng tầng RVQ (n_q = 4 trong EnCodec)
            codebook_size=1024,  # Kích thước mỗi codebook trong EnCodec
            d_model=768,  # Embedding dimension của Transformer
            nhead=12,  # Số lượng Multi-head Attention
            num_layers=12,  # Số lớp Transformer Decoder Layer
            max_seq_len=1500,  # Độ dài chuỗi token tối đa
            t5_model_name="t5-base"
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.d_model = d_model

        # 1. Text Encoder & Conditioning
        self.text_conditioner = TextConditioner(t5_model_name=t5_model_name, d_model=d_model)

        # 2. Audio Token Embeddings (0 -> codebook_size-1: audio code, codebook_size: BOS token)
        self.bos_token_id = codebook_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_codebooks)
        ])

        # Positional Encoding cho chuỗi thời gian Audio
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # 3. Transformer Decoder Backbone
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. Output Projection Heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_codebooks)
        ])

        # 5. EnCodec Model để decode token ra audio waveform (Khởi tạo lazy khi gọi)
        self.encodec_decoder = None

    def _shift_right(self, audio_tokens):
        """
        Teacher forcing chuẩn: [BOS, token_0, ..., token_(T-2)].
        Decoder không bao giờ được nhìn thấy token mà nó phải dự đoán.
        """
        B, K, T = audio_tokens.shape
        shifted_tokens = torch.full(
            (B, K, T), self.bos_token_id,
            dtype=torch.long,
            device=audio_tokens.device,
        )
        if T > 1:
            shifted_tokens[:, :, 1:] = audio_tokens[:, :, :-1]
        return shifted_tokens

    def forward(self, audio_tokens, text_list):
        """
        Huấn luyện Forward Pass
        """
        device = audio_tokens.device
        B, K, T = audio_tokens.shape

        # 1. Trích xuất Text Condition Embeddings
        text_embeds, text_mask = self.text_conditioner(text_list, device)

        # 2. Dịch phải input để loại target leakage
        input_tokens = self._shift_right(audio_tokens)
        seq_len = input_tokens.shape[-1]

        # 3. Tổng hợp Embedding từ K tầng Codebook
        audio_embeds = torch.zeros(
            B, seq_len, self.d_model, device=device, dtype=self.pos_embedding.dtype
        )
        for k in range(K):
            audio_embeds = audio_embeds + self.embeddings[k](input_tokens[:, k, :])

        audio_embeds = audio_embeds + self.pos_embedding[:, :seq_len, :]

        # 4. Causal Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        memory_key_padding_mask = (text_mask == 0)

        # 5. Transformer Decoder
        output_embeds = self.transformer_decoder(
            tgt=audio_embeds,
            memory=text_embeds,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # 6. Predict Logits
        logits = torch.stack([head(output_embeds) for head in self.heads], dim=1)

        return logits

    @torch.no_grad()
    def generate(self, text_list, max_steps=750, temperature=1.0, top_k=250, cfg_scale=3.0):
        """
        Hàm Inference (Sinh nhạc từ Text Prompt) hỗ trợ Classifier-Free Guidance (CFG).
        cfg_scale: Độ mạnh bám theo prompt (Thường chọn 3.0 -> 5.0).
        """
        self.eval()
        device = next(self.parameters()).device
        B = len(text_list)
        K = self.num_codebooks

        # Encode both CFG branches together so padding gives them equal length.
        uncond_texts = [""] * B
        combined_text_embeds, combined_text_mask = self.text_conditioner(
            text_list + uncond_texts, device
        )
        memory_key_padding_mask = (combined_text_mask == 0)

        # Khởi tạo token ban đầu
        generated_tokens = torch.full((B, K, 1), self.bos_token_id, dtype=torch.long, device=device)

        for step in range(max_steps):
            seq_len = generated_tokens.shape[-1]

            if seq_len > self.pos_embedding.shape[1]:
                break

            # Embeddings cho cả 2 luồng (Cond & Uncond)
            audio_embeds = torch.zeros(
                B, seq_len, self.d_model, device=device, dtype=self.pos_embedding.dtype
            )
            for k in range(K):
                audio_embeds = audio_embeds + self.embeddings[k](generated_tokens[:, k, :])
            audio_embeds = audio_embeds + self.pos_embedding[:, :seq_len, :]

            # Đúp đôi batch cho CFG
            combined_audio_embeds = torch.cat([audio_embeds, audio_embeds], dim=0)

            # Forward qua Decoder
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            output_embeds = self.transformer_decoder(
                tgt=combined_audio_embeds,
                memory=combined_text_embeds,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

            # Lấy logits bước cuối
            next_logits = torch.stack([head(output_embeds[:, -1, :]) for head in self.heads],
                                      dim=1)  # [2B, K, Codebook_size]

            # Tách logits thành luồng Cond và Uncond
            cond_logits, uncond_logits = next_logits.chunk(2, dim=0)

            # Áp dụng Classifier-Free Guidance (CFG Formula)
            guided_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            guided_logits = guided_logits / max(temperature, 1e-5)

            # Top-K Sampling
            if top_k > 0:
                v, _ = torch.topk(guided_logits, min(top_k, guided_logits.size(-1)))
                pivot = v[..., -1:]
                guided_logits = torch.where(guided_logits < pivot, torch.tensor(-float("Inf"), device=device),
                                            guided_logits)

            probs = torch.softmax(guided_logits, dim=-1)

            # Sample token tiếp theo
            next_tokens = torch.zeros(B, K, 1, dtype=torch.long, device=device)
            for k in range(K):
                next_tokens[:, k, 0] = torch.multinomial(probs[:, k, :], num_samples=1).squeeze(-1)

            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)

        return generated_tokens[:, :, 1:]  # Bỏ BOS token

    @torch.no_grad()
    def tokens_to_audio(self, tokens):
        """
        BỔ SUNG MỚI: Decode ma trận Audio Tokens thành sóng âm thanh (.wav Waveform)
        tokens shape: [B, K, T]
        """
        device = tokens.device
        if self.encodec_decoder is None:
            self.encodec_decoder = EncodecModel.encodec_model_24khz().to(device)
            self.encodec_decoder.set_target_bandwidth(3.0)
            self.encodec_decoder.eval()

        # EnCodec yêu cầu input theo dạng list của tuple (code, scale)
        encoded_frames = [(tokens.to(device), None)]
        waveform = self.encodec_decoder.decode(encoded_frames)
        return waveform  # [B, Channels, Samples] - 24kHz Audio Waveform


# ==========================================
# Code kiểm thử Mô hình (Run Test)
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy test mô hình trên thiết bị: {device}")

    model = MusicTransformerLM(
        num_codebooks=4,
        codebook_size=1024,
        d_model=512,
        nhead=8,
        num_layers=6
    ).to(device)

    dummy_texts = ["A 90s rock song with heavy drums", "Chill lofi beats for studying"]

    print("\n--- TEST MUSIC GENERATION WITH CFG ---")
    generated_tokens = model.generate(dummy_texts, max_steps=50, cfg_scale=3.0)
    print("Generated Tokens Shape:", generated_tokens.shape)  # [2, 4, 50]

    print("\n--- TEST DECODE TO AUDIO WAVEFORM ---")
    waveform = model.tokens_to_audio(generated_tokens)
    print("Decoded Waveform Shape:", waveform.shape)  # Ex: [2, 1, 16000]
