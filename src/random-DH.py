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
    **在提出问题或进行推理时，请跳出固定思维模式，主动识别并思考所有看似细微但可能具有关键意义的线索（例如，某个模糊的代词指向、某个看似无关的物品或行为），这些都可能是揭示真相的重要突破口。**
    ------------------------
    仍然需要探索时，严格按照下面的 JSON 格式输出：
    {{
      "correlation_questions": [
        {{
          "dimension":"<问题是为了解决哪一个维度的>"
          "questions":"<你要探索的问题>"
        }}
      ],
      "finish": 0
    }}
    """
    prompt = prompt_head + prompt_tail1 + prompt_tail2
    if step == 0: prompt = f"""你提出的问题以及答案{answer},继续探索"""
    conversation_history.append({"role": "user", "content": prompt})
    llm_output = llm2.invoke(conversation_history).content
    conversation_history.append({"role": "assistant", "content": llm_output})
    parsed = json.loads(llm_output)
    return parsed

def decompose_obstacle(obstacle):
    prompt = f"""
    你是一名资深侦探，擅长理解问题，为了解决复杂笼统的问题，你的任务是将这些复杂问题**分解**成一系列清晰、可调查的子问题。

    ---
    案件信息：{case_info}
    ---
    **核心任务：** 针对以下总问题 `{obstacle}`，请你将其分解为**最多3个**独立的、具体且可供后续侦查人员调查的子问题。
    最后一个问题必须是原始问题

    ---
    **输出格式要求：** 严格按照下面的 JSON 格式输出。
    {{
       "parent_q": "{obstacle}",
       "sub_q": [
       "第一个需要调查的子问题是什么？",
       "第二个需要调查的子问题是什么？",
       "{obstacle}",
       ],
       "key": 1
    }}
    """
    llm_output = llm2.invoke([HumanMessage(content=prompt)]).content
    parsed = json.loads(llm_output)
    sub_answer = []
    if parsed.get("key", 0) == 1:
        sub_questions = parsed.get("sub_q", [])

        # 使用线程池并行处理所有子问题
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_question = {
                executor.submit(answer_sub_question, q, 1): q
                for q in sub_questions
            }

            for future in concurrent.futures.as_completed(future_to_question):
                result = future.result()
                sub_answer.append(result)
        return sub_answer

def answer_sub_question(obstacle, depth):
    prompt = f"""
    在案件调查中，某些关键信息（例如：精确的作案过程、嫌疑人未暴露的动机）不可能被直接记录。
    你的任务是根据现有案件信息，运用你的专业侦探直觉和逻辑推理，对问题 {obstacle} 给出最高质的假设答案。

    案件信息：{case_info}
    ----------
    重要⚠️：常识性侦探准则：
    1.时间线核查:  
        受害人症状发作的时间点，与嫌疑人接触或投毒的时间点，存在矛盾。 
    2.空间位置核查: 
        如果作案发生时有多个人在场且具备观察或干预的能力，那么单一凶手进行的作案行为其机会几乎为零。
        在这种情况下，除非有明确证据表明凶手有能力制造或利用独处/无干扰的环境（例如：制造混乱、引开他人、受害人高度虚弱或分心、环境极度昏暗等），否则应优先排除在此类环境下单独作案的可能性。
    3.行为模式与物理定律核查:
        检查证词和事件描述是否符合基本的物理定律（例如：一个人不可能在短时间内从A地瞬间移动到B地，除非有交通工具且距离合理）。
    ----------
    ⚠️ 你下面的的next_step绝对不可以和parent_q相同导致浪费资源
    严格按照下面的 JSON 格式输出：
    {{
       parent_q:<当前解决的问题>,
       answer_q:<高质推理假设的答案/无相关信息>,
       next_step:进一步调查的信息/无>,
       key:2
    }}
    """
    llm_output = llm2.invoke([HumanMessage(content=prompt)]).content
    parsed = json.loads(llm_output)
    if (parsed.get("key", 0) == 2 and
            parsed.get("answer_q", "") != "无相关信息" and
            parsed.get("next_step", "") != "无"
            and depth <= 1):
        next_answer = answer_sub_question(parsed.get("next_step", ""), depth + 1)
        return f"""子问题：{parsed.get("parent_q", "")} 答案：{parsed.get("answer_q", "")}\n{next_answer}"""
    else:
        return f"""子问题：{parsed.get("parent_q", "")} 答案：{parsed.get("answer_q", "")}"""




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
        for q in res1.get("correlation_questions", []):
            q = q.get("questions", "")
            answer = decompose_obstacle(q)
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
