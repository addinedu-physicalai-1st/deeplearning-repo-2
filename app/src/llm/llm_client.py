import ollama
import json

class LLMClient:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name

    def generate_feedback(self, session_data: dict) -> str:
        """
        LLM에게 JSON 포맷으로 격려 멘트와 구체적인 피드백을 요청합니다.
        """
        duration_min = session_data['duration'] // 60
        score = session_data['focus_score']
        distractions = session_data['distract_cnt']
        model_type = session_data['model_type']
        
        # 프롬프트 설계: JSON 출력을 강제함
        prompt = (
            f"당신은 업무 생산성 코치입니다. 다음 세션 데이터를 분석해주세요.\n"
            f"- 집중 시간: {duration_min}분\n"
            f"- 집중 점수: {score}점\n"
            f"- 산만 횟수: {distractions}회\n"
            f"- 감지 모델: {model_type}\n\n"
            f"반드시 아래 JSON 형식으로만 응답하세요. 다른 말은 하지 마세요.\n"
            f"{{\n"
            f"  \"comment\": \"사용자의 감정을 고려한 따뜻한 격려나 위로의 말 (1~2문장)\",\n"
            f"  \"feedback\": \"데이터에 기반한 구체적이고 실천 가능한 행동 교정 팁 (개조식으로 3가지)\"\n"
            f"}}"
        )

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': 'You are a helpful coach. Output ONLY valid JSON.'},
                {'role': 'user', 'content': prompt},
            ])
            
            content = response['message']['content']
            
            # JSON 파싱 검증 (파싱 실패 시 텍스트 그대로 반환 방지)
            # 가끔 LLM이 ```json ... ``` 형태로 줄 때가 있어 처리
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            return content.strip()
            
        except Exception as e:
            # 에러 발생 시 fallback JSON 반환
            error_json = {
                "comment": "분석 중 오류가 발생했습니다.",
                "feedback": f"Ollama 연결 상태를 확인해주세요. ({str(e)})"
            }
            return json.dumps(error_json, ensure_ascii=False)
