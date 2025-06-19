import json

from langchain.schema import HumanMessage

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import warnings
import concurrent.futures

warnings.filterwarnings("ignore", category=UserWarning)

llm2 = ChatOpenAI(
    model="",
    temperature=0.0,
    max_tokens=8192,
    openai_api_base="",
    openai_api_key="",
    streaming=True,
    response_format={
        'type': 'json_object'
    },
    callbacks=[StreamingStdOutCallbackHandler()]
)
llm = ChatOpenAI(
    model="",
    temperature=0.0,
    max_tokens=8192,
    openai_api_base="",
    openai_api_key="",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

with open(r"../dataset/pjmsa/role.json", 'r', encoding='utf-8') as f:
    role = json.load(f)
with open(r"../dataset/pjmsa/evidence.json", 'r', encoding='utf-8') as f:
    evidence = json.load(f)
with open(r"../dataset/pjmsa/event.json", 'r', encoding='utf-8') as f:
    event = json.load(f)


def identify_obstacle(answer, step):
    prompt_head = f"""
    最终任务：找出真相
    完成最终任务需要的工作如下： 
       提出重建、还原真相过程中的障碍。
       你能通过提出引导性的问题来解决你的障碍，你的引导性问题必须能准确犀利的帮你理清整体事件
    ⚠️ 注意：
    1. “障碍”：还原完整犯罪过程时，当前信息不足以解释或推导清楚的关键环节；
    2. 重点关注“如何作案、如何解释关键现象”，如何解释作案手法的机制。
    
    ------------------------
    案件信息：{case_info}

    """
    prompt_tail2 = f"""
    ------------------------
    该维度探索结束时，严格按照下面的 JSON 格式输出：
    {{
      "finally_completely_suspect_modus_operandi": [
          {{
            "suspect":"<最终锁定的真凶>",
            "motive": "<直接导火索(事件描述: 严格原文; 逻辑解释: 逻辑严密的为什么该事件发生以及如何具体导致杀意的产生(重要)) + 深层驱动因素>"
            "modus_operandi":"<最终真凶的完整且无矛盾的作案手法与过程>",
            "key_clue": [
            {{
               "clue_description": "<对原始线索的概括与提炼>",
               "implication": "<这条线索在推理中的意义，它如何帮助排除/锁定嫌疑人或支持作案手法>"
            }},
            // 更多关键线索
            ]
          }},
        ...
      ],
      exclude_suspect_role_reason:"<排除的嫌疑人以及原因>"
      "finish": 4
    }}
    """
    prompt_tail1 = f"""
    你能通过提出相关问题来获得信息协助你完成任务。
    ------------------------
    仍然需要探索时，严格按照下面的 JSON 格式输出：
    {{
      "correlation_questions": [
        {{
          "dimension":"<问题是为了解决哪一个维度的>"
          "questions":"<你要探索的问题>"
          "answers":"<你认为的问题的答案>"
        }}
      ],
      "finish": 0
    }}
    """

    prompt = prompt_head  + prompt_tail1 + prompt_tail2
    if step == 0: prompt = f"""你提出的问题以及答案{answer},继续探索"""
    conversation_history.append({"role": "user", "content": prompt})
    llm_output = llm2.invoke(conversation_history).content
    conversation_history.append({"role": "assistant", "content": llm_output})
    parsed = json.loads(llm_output)
    return parsed



if __name__ == "__main__":
    case_info = f""" -角色信息:{role} -物品信息: {evidence}  -事件信息: {event}"""
    options = []
    big_options = []
    temp_options = []
    raw_options=[]
    conversation_history = []
    step = 3
    while step != 7 - 3:
        res1 = identify_obstacle(temp_options, step)
        step = res1.get("finish", 0)
        if step == 7 - 3:
            big_options.append(res1.get("finally_completely_suspect_modus_operandi", []))
            break
        for ques in res1.get("correlation_questions", []):
            q = ques.get("questions", "")
            answer = ques.get("answers", "")
            temp_options.append({
                "obstacle": q,
                "answer": answer
            })
        print(temp_options)
        options.append(temp_options)

    with open(r"../database/big_options.json", 'w', encoding='utf-8') as f:
        json.dump(big_options, f, ensure_ascii=False, indent=4)

    with open(r"../database/options.json", 'w', encoding='utf-8') as f:
        json.dump(options, f, ensure_ascii=False, indent=4)



