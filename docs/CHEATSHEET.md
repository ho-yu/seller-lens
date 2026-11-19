# 핵심 코드 모음

> 실무에서 매번 쓰는 코드만 강마다 쌓는다. 가리고 다시 쓸 수 있으면 된다.
> 강을 시작할 때 여기서 빈칸 퀴즈 3문항이 나온다 (지난 강 2개, 더 예전 것 1개). 마무리 때 Claude가 그날 🧠을 아래에 붙인다.

## Day05 · LLM API 호출 (1과목 5장)

```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama를 OpenAI SDK로 부른다
messages = [
    {"role": "system", "content": system_prompt},  # 규칙
    {"role": "user", "content": review},           # 이번 입력
]
response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
answer = response.choices[0].message.content.strip()  # 답 문자열만 꺼낸다
```

- `temperature=0`은 같은 입력에 같은 답을 내게 한다. 분류와 평가에 쓴다.

## Day07 · Few-shot과 평가 (1과목 6장)

```python
messages += [{"role": "user", "content": example_review}, {"role": "assistant", "content": answer}]  # 예시 한 쌍

recall = tp / (tp + fn)     # 재현율: 진짜 부정 중 잡은 비율
precision = tp / (tp + fp)  # 정밀도: 부정이라 한 것 중 진짜 비율
```

- 예시로 보여 준 리뷰는 평가셋에 넣지 않는다.
- 정확도는 "전부 한쪽으로 답했을 때"의 기준선과 비교해서 읽는다.

## Day08 · Structured Output과 Pydantic 검증 (1과목 7장)

```python
class ReviewTag(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")  # 자동 형 변환과 모르는 필드를 막는다
    sentiment: Literal["부정", "중립", "긍정"]                # 허용값은 Literal
    urgency: int = Field(ge=1, le=3)                         # 범위는 Field

try:
    tag = ReviewTag.model_validate_json(raw)                 # 문자열 → 검증된 객체
except ValidationError as error:
    print(error.errors())                                    # 오류마다 loc(위치)·type·msg
```

- 스키마는 `response_format`으로 모델에 넘기고(`ReviewTag.model_json_schema()`), 받은 답도 같은 클래스로 검증한다.
- 실패를 가짜 값으로 메우지 않는다. `(None, 오류 목록)`을 돌려준다.
