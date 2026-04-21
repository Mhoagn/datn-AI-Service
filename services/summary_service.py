"""
Text summarization service using Qwen
"""
from models import model_manager
from config import MAX_INPUT_LENGTH, MAX_SUMMARY_LENGTH

class SummaryService:
    """Handle text summarization"""
    
    @staticmethod
    def summarize(text: str, duration_seconds: float = 0, max_length: int = MAX_SUMMARY_LENGTH) -> str:
        """
        Summarize text using Qwen 2.5 1.5B Instruct
        Returns formatted bullet points for easy parsing
        
        Args:
            text: Text to summarize
            duration_seconds: Video duration in seconds (for determining number of points)
            max_length: Max length of summary
        """
        print("Tóm tắt nội dung với Qwen 2.5 1.5B Instruct...")
        
        if not text or len(text.strip()) < 10:
            print("Văn bản quá ngắn hoặc rỗng, bỏ qua tóm tắt")
            return "1. Không đủ nội dung để tóm tắt"
        
        # Check if text is an error message
        if text.startswith("[Loi") or text.startswith("[Khong"):
            print(f"⚠️ Text is an error message: {text}")
            return "1. Lỗi khi xử lý transcript, không thể tóm tắt"
        
        if len(text) > MAX_INPUT_LENGTH:
            text = text[:MAX_INPUT_LENGTH] + "..."
        
        # Determine number of summary points based on duration
        # < 15 minutes (900 seconds): 3 points
        # >= 15 minutes: 7 points
        duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 0
        num_points = 3 if duration_minutes < 15 else 7
        
        print(f"📊 Duration: {duration_minutes:.1f} minutes → {num_points} điểm chính")
        
        prompt = f"""Bạn là trợ lý AI chuyên nghiệp, nhiệm vụ của bạn là tóm tắt nội dung cuộc họp thành các ý chính.

Nội dung cuộc họp:
{text}

Hãy tóm tắt thành CHÍNH XÁC {num_points} điểm chính, mỗi điểm trên một dòng, bắt đầu bằng số thứ tự. Ví dụ:
1. Điểm thứ nhất
2. Điểm thứ hai
3. Điểm thứ ba

Tóm tắt:"""

        messages = [
            {"role": "system", "content": "Bạn là trợ lý AI tóm tắt cuộc họp chuyên nghiệp. Luôn trả lời bằng tiếng Việt và format theo dạng danh sách đánh số."},
            {"role": "user", "content": prompt}
        ]
        
        tokenizer = model_manager.get_tokenizer()
        model = model_manager.get_qwen_model()
        
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        try:
            summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            summary = summary.strip()
            
            print(f"\n{'='*60}")
            print(f"📄 QWEN OUTPUT (Raw):")
            print(f"{'='*60}")
            print(summary)
            print(f"{'='*60}\n")
            
            lines = [line.strip() for line in summary.split('\n') if line.strip()]
            formatted_lines = []
            
            for i, line in enumerate(lines, 1):
                if not line[0].isdigit():
                    formatted_lines.append(f"{i}. {line}")
                else:
                    formatted_lines.append(line)
            
            result = '\n'.join(formatted_lines) if formatted_lines else summary
            
            print(f"Tóm tắt hoàn tất: {len(formatted_lines)} điểm chính")
            return result
            
        except Exception as e:
            print(f"Lỗi khi tạo tóm tắt: {str(e)}")
            return "1. Lỗi khi tạo tóm tắt"
