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


def information_fusion_annotation():
    prompt = f"""
    你是一名资深的案件分析师，擅长整合信息并找出案件中的疑点。我将提供三份案件相关信息：物品清单、事件时间线和人物角色描述。请你仔细阅读并完成以下任务：

    1.  **整合并扩充事件时间线：** 以事件时间线为主干，将物品清单和人物角色描述中的相关信息融入到对应的时间节点。
        * **物品信息补充：** 将事件时间线出现物品的详细描述、发现位置以及任何相关信息补充到对应的事件描述中。
        * **人物信息关联分析：** 将人物的性格与他们在事件时间线中的行为进行对比和质疑  2.将人物角色的证词与他们在事件时间线中的行为进行对比和融合。如果证词与事件描述存在印证、矛盾。 
    2.  **添加案件疑点批注（基于常识推理）：
           ** 整合后的案件信息中存在可疑、不合常理、逻辑冲突或需要进一步调查的地方，使用"【批注：...】"的形式进行标注。
           ** 这些批注应基于常识和推理，跨越时间线，并能指出潜在的矛盾、未解释的现象、可能存在的误导信息或值得深挖的线索。
    3.批注物证信息，在事件时间线结束之后，对物证信息进行批注

    ----------------------
    要求：
    1.关于事件的描述、物证必须是严格的原文，不能少任何一条时间线  且任何一条时间线的里面的任何内容也不能少
    2.每条时间线、物证信息后面必须有批注
    ----------------------

    **以下是待整合的案件信息：**
    事件信息:{event}
    角色信息:{role}
    物品信息:{evidence}

    关于时间线的相关信息输出完毕时候立刻结束，不要输出总结性的信息，比如 **关键疑点总结** 、**建议进一步调查**  
    不用严格按照json格式
    """
    llm_output = llm.invoke([HumanMessage(content=prompt)]).content
    return llm_output

def structured_reasoning(info, answer_info, answer, step, n):
    prompt_head = f"""
    最终任务：找出真相
    完成最终任务需要的工作如下： 
    ------------------------
    案件信息：{case_info}
    """
    if step == 1:
        prompt_body = f"""
        **你的角色设定：** 你现在是一名**深度动机分析师**。你的唯一职责是**对每个嫌疑人进行心理和行为层面的动机剖析**。在此阶段，你**不关心也无需分析任何作案手法或过程**，只专注于“**为什么会发生谋杀？**”以及“**谁有最充分的杀人动机？**”
        1.**作案动机分析**：严格按照下面的步骤进行分析，必须严格根据下面的进行思考提问，不能想当然。
            对每个嫌疑人，深入分析使其谋杀具体动机(直接导火索)。
            * **直接导火索：** * **核心定义：** 指的是一个**明确的、在嫌疑人决定实施谋杀行为之前，被嫌疑人直接感知到(私下的不算)并深刻影响了其态度的关键事件或信息**。这个事件必须**直接促使嫌疑人产生了杀人意图或做出了杀人决定，是其杀人意图从无到有的那个转折点**。
            * **判断依据：** 直接导火索必须是**导致嫌疑人内心从“没有杀人意图”转变为“产生杀人意图”的那个核心刺激点**。
        2.嫌疑人排除：
            在完成上述动机分析后，根据动机是否足够强烈和具有说服力，排除部分嫌疑人。排除依据：(如果某嫌疑人缺乏足够强烈的、令人信服的谋杀动机。如果其动机在逻辑上不足以支撑谋杀行为。)
        """

        prompt_tail2 = f"""
        ------------------------
        该维度探索结束时，根据你的知识库(案件信息以及探索内容)，给出下面的完整的，不要敷衍的答案。
        严格按照下面的 JSON 格式输出：
        {{
          "remain_suspect_role_motive": [
            {{
              "suspect":"<剩下的嫌疑人>",
              "motive": "<直接导火索( 严格原文描述以及如何具体导致杀意的产生(重要)) + 深层驱动因素>"
            }},
            ...
          ],
          exclude_suspect_role_reason:"<排除的嫌疑人以及原因>"
          "finish": 6
        }}
        """
    elif step == 2:
        prompt_body = f"""
        第1个维度得到的嫌疑人以及动机信息如下：{info}
        ⚠️*你接下来的嫌疑人只能从这里面找，不能自己添加。*
        ------------------------
        **你的角色设定：** 你现在是一名**犯罪现场行为分析师**。你的唯一职责是**纯粹基于逻辑和物理可能性，重建和评估作案手法**。你**不负责分析或推测作案动机**，也**不考虑嫌疑人的心理状态**。只专注于“**凶手是如何做到的？**”和“**谁能做到？**”
        1.* **探索完整作案手法**
            * **获取途径与时机：** 凶手获得作案工具的方式、途径、以及其使用过程必须符合逻辑。这包括但不限于：
                * **位置可达性：** 作案工具的获得必须与嫌疑人当时所处的位置或作案工具存放的位置相符。如果作案工具在A地，而嫌疑人在作案时始终身处B地且没有合理机会前往A地，那么他就不可能获取该工具。
                * **时间窗口：** 嫌疑人必须有足够的时间和机会去获取作案工具。例如，如果作案工具需要特定时间才能获得，而嫌疑人在该时间段内没有行动自由，则应排除。
            * **独立操作机会：** 嫌疑人是否具备**单独操作作案工具**的机会？
                * 如果作案手法需要单独行动（如秘密下毒、单独放置陷阱等），而嫌疑人在关键作案时间点**始终与他人抱团行动，没有独处机会**，则他无法实施此类作案，应将其排除。
                * 反之，如果作案手法允许协作或在人群中实施（如制造混乱中投掷物品），则需要进一步考虑。
        2.* **探索排除与剩余嫌疑人：
             * 根据上述对作案手法、作案工具获取的时空限制、以及独立操作机会的深入分析，请具体说明哪些嫌疑人可以被排除？
             * 排除这些嫌疑人的**具体依据是什么？** 请详细说明他们不符合作案手法的逻辑之处。
             * 现在还剩余哪些嫌疑人？
        """

        prompt_tail2 = f"""
        ------------------------
        该维度探索结束时，根据你的知识库(案件信息以及探索内容)，给出下面的完整的，不要敷衍的答案。
        严格按照下面的 JSON 格式输出：
        {{
          "remain_suspect_role_modus_operandi": [
            {{
              "suspect":"<剩下的当前嫌疑人>",
              "motive": "<原来的动机 >"
              "modus_operandi":"<以时间线来介绍的，非常完整符的作案手法与过程>"
            }},
            ...
          ],
          exclude_suspect_role_reason:"<排除的嫌疑人以及原因>"
          "finish": 5
        }}
        """
    else:
        prompt_body = f"""
        第3个维度得到的嫌疑人以及动机信息如下：{info}
        ⚠️*你接下来的嫌疑人只能从这里面找，不能自己添加。*
        ------------------------
        **你的角色设定：** 你现在是一名**专业案件信息审计师**，你的核心职责是进行**“地毯式无遗漏审查”**。你不是在重复已知信息，而是要**主动发现并标记出案件全貌{case_info}中所有被忽视的“空白点”和“异常点”**。

        * 你的目标是**穷尽所有在`answer_info`中“没有被提及”、“没有被充分展开讨论”或“被完全忽略”的细节、对话、行为、时间点或任何潜在线索**。
        * **绝对禁止重复** 探索内容{json.dumps(answer_info, ensure_ascii=False, indent=2)}中已有的任何信息。你所识别的每一个“疑点”都必须是**全新的发现**，即`answer_info`中从未出现过或未被深入分析的。
        * 请特别关注那些**不寻常的对话、反常的人物行为、任何看起来与案件无关却可能暗示深层关系的细节、以及时间线上可能存在的微妙异常**。

        **3. 全面探索疑点：**
            * 3.1 **全面疑点探索：** 在这个阶段，你需要跳出已知的、显而易见的线索，主动识别并深入剖析**此前被忽略的细节**。你的目标是**穷尽所有可能的疑点，不放过任何一个死角。**
            * 3.2 **疑点对理论的补充与修正：** 基于这些新发现的疑点，你需要**补充或深化**现有嫌疑人的**作案动机**，以及**修正或完善**推测的**作案手法与过程**。
            * 3.2 持续进行探索，直到你确信所有相关信息都已被充分挖掘，对案件的理解达到前所未有的深度和全面性，**并且你已经能确定真正的凶手**
        """
        prompt_tail2 = f"""
        ------------------------
        该维度探索结束时，严格按照下面的 JSON 格式输出：
        {{
          "finally_completely_suspect_modus_operandi": 
              {{
                "suspect":"<最终锁定的一个真凶>",
                "motive": "<经过你补充过后的  直接导火索(事件描述: 严格原文; 逻辑解释: 逻辑严密的为什么该事件发生以及如何具体导致杀意的产生(重要)) + 深层驱动因素  ，非常细节和完整，丝毫不敷衍>"
                "modus_operandi":"<经过你补充过后的  最终真凶的完整且无矛盾的作案手法与过程，非常细节和完整，丝毫不敷衍>"
              }},
          exclude_suspect_role_reason:"<排除的嫌疑人以及原因>",
          "finish": 4
        }}
        """
    prompt_tail1 = f"""
    ------------------------
    在提出问题或进行推理时，你能通过提出相关问题来获得信息协助你完成任务，
    重点：为了防止重复探索，你提出的questions绝对严格的不能与已提及的问题{already_question}在语义上重复。
    ------------------------
    当你需要探索时，严格按照下面的 JSON 格式输出：
    {{
      "correlation_questions": [
        {{
          "questions":"<你要探索的问题>",
        }}
      ],
      "finish": 0
    }}
    """

    if step == 0 and n == 2:
        prompt = f"""你提出的问题以及答案{answer},你不能继续探索了，请输出答案"""
    elif step == 0:
        prompt = f""" 你提出的问题以及答案{answer},继续探索"""
    else:
        prompt = prompt_head + prompt_body + prompt_tail1 + prompt_tail2
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
    #    ----------
    # 案件信息{case_info}
    llm_output = llm2.invoke([HumanMessage(content=prompt)]).content
    parsed = json.loads(llm_output)
    sub_answer = []
    if parsed.get("key", 0) == 1:
        sub_questions = parsed.get("sub_q", [])

        # 使用线程池并行处理所有子问题
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有任务到线程池
            future_to_question = {
                executor.submit(heuristic_hypothesis, q, 1): q
                for q in sub_questions
            }

            for future in concurrent.futures.as_completed(future_to_question):
                # q = future_to_question[future]
                result = future.result()
                sub_answer.append(result)
        # return solve(obstacle,sub_answer,before_info)
        return sub_answer

def heuristic_hypothesis(obstacle, depth):
    prompt = f"""
    在案件调查中，某些关键信息（例如：精确的作案过程、嫌疑人未暴露的动机）不可能被直接记录。
    你的任务是根据现有案件信息，运用你的专业侦探直觉和逻辑推理，对问题 {obstacle} 给出最高质的假设答案。

    案件信息：{case_info}
    你需要先从案件信息找出问题所在上下文，然后根据上下文和背景信息，进行合理判断和假设。
    ----------
    重要⚠️：常识性侦探准则：
    1.时间线核查:  
        受害人症状发作的时间点，与嫌疑人接触或投毒的时间点，存在矛盾。 
    2.空间位置核查: 
        如果作案发生时有多个人在场且具备观察或干预的能力，那么单一凶手进行的作案行为其机会几乎为零。
        除非有明确证据表明凶手有能力制造或利用独处/无干扰的环境（例如：制造混乱、引开他人、受害人高度虚弱或分心、环境极度昏暗等），否则应优先排除在此类环境下单独作案的可能性。
    3.行为模式与物理定律核查:
        检查证词和事件描述是否符合基本的物理定律（例如：一个人不可能在短时间内从A地瞬间移动到B地，除非有交通工具且距离合理）。
    ----------
    ⚠️ 你下面的的next_step绝对不可以和parent_q相同导致浪费资源
    严格按照下面的 JSON 格式输出：
    {{
       parent_q:<当前解决的问题>,
       answer_q:<高质推理假设的答案/无相关信息>,
       next_step:<进一步调查的信息/无>,
       key:2
    }}
    """
    llm_output = llm2.invoke([HumanMessage(content=prompt)]).content
    parsed = json.loads(llm_output)
    if (parsed.get("key", 0) == 2 and
            parsed.get("answer_q", "") != "无相关信息" and
            parsed.get("next_step", "") != "无"
            and depth <= 1):
        next_answer = heuristic_hypothesis(parsed.get("next_step", ""), depth + 1)
        return f"""子问题：{parsed.get("parent_q", "")} 答案：{parsed.get("answer_q", "")}\n{next_answer}"""
    else:
        return f"""子问题：{parsed.get("parent_q", "")} 答案：{parsed.get("answer_q", "")}"""


if __name__ == "__main__":
    event=information_fusion_annotation()
    case_info = f"""-事件信息: {event} -角色信息:{role} -物品信息: {evidence}"""
    options = []
    big_options = []
    already_question = []
    keys = ["remain_suspect_role_motive", "remain_suspect_role_modus_operandi",
            "finally_completely_suspect_modus_operandi"]

    for i in range(1, 4):
        n = 0
        temp_options = []
        conversation_history = []
        step = i
        bg = big_options[i - 2] if i > 1 else ""
        while step != 7 - i:
            n = n + 1
            print(bg)
            res1 = structured_reasoning(bg, options, temp_options, step, n)
            step = res1.get("finish", 0)
            if step == 7 - i:
                big_options.append(res1.get(keys[i - 1], []))
                break
            correlation_questions = res1.get("correlation_questions", [])
            already_question.append(correlation_questions)
            # --- 多线程并行处理 ---
            if correlation_questions:
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_question = {
                        executor.submit(decompose_obstacle, q_item.get("questions", "")): q_item.get(
                            "questions", "")
                        for q_item in correlation_questions
                    }
                    # 收集已完成任务的结果
                    for future in concurrent.futures.as_completed(future_to_question):
                        original_question_text = future_to_question[future]  # 获取原始的问题文本
                        answer_result = future.result()
                        temp_options.append({
                            "obstacle": original_question_text,
                            "answer": answer_result  # 这里 'answer' 是 summarize 后的字符串
                        })
            print(temp_options)
            options.append(temp_options)

    with open(r"../database/big_options.json", 'w', encoding='utf-8') as f:
        json.dump(big_options, f, ensure_ascii=False, indent=4)

    with open(r"../database/options.json", 'w', encoding='utf-8') as f:
        json.dump(options, f, ensure_ascii=False, indent=4)


