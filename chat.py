from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 修正导入
#from langchain.tools.render import format_tool_to_openai_function  # 关键新增
from langchain_core.utils.function_calling import convert_to_openai_function
import requests
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

@tool
def get_weather(city: str) -> str:
    """查询城市的天气（使用seniverse） """
    weather_api_url = 'https://api.seniverse.com/v3/weather/now.json'
    params = {
        'key': 'S_etVAfPVYlMLReWC', 
        'location': city,
        'language': 'zh-Hans',
        'unit': 'c'
    }
    response = requests.get(weather_api_url, params=params)
    weather_data = response.json()
    
    if 'results' in weather_data and weather_data['results']:
        result = weather_data['results'][0]['now']
        return f"{city}当前天气：{result['text']}，温度：{result['temperature']}°C"
    else:
        return "无法获取{city}天气信息，请稍后再试。"

@tool
def calculate_string_length(text: str) -> int:
    """计算输入字符串的长度（自定义工具示例）"""
    print(f"[DEBUG] 调用 calculate_string_length，输入: {text}")
    return len(text)

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """获取指定时区的当前时间（模拟实现）"""
    print(f"[DEBUG] 调用 get_current_time，时区: {timezone}")
    from datetime import datetime, timezone as tz
    now = datetime.now(tz.utc).astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

llm = ChatDeepSeek(
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-e4e5862ddc5046f496cdb306775085cb",
    # other params...
)

tools = [calculate_string_length, get_current_time, get_weather]
system_prompt = """
你是一个智能助手，可以调用工具解决问题。
请严格按照以下规则工作：
1. 仔细分析用户需求，选择最合适的工具
2. 如果用户需求需要多个工具协作，分步骤执行
3. 如果无法通过工具解决问题，礼貌说明原因
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),      # 元组格式
    ("human", "{input}"),           # 元组格式
    MessagesPlaceholder(variable_name="agent_scratchpad")  # 直接作为列表项
])

agent = create_tool_calling_agent(llm, tools, prompt)
formatted_tools = [convert_to_openai_function(t) for t in tools]

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({
    #"input": "大连今天的天气怎么样？",
    "input": "'Hello World' 的长度是多少？",
    "tools": formatted_tools  # 显式传入工具描述
})
print(result["output"])

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True
# )

# print("=== 测试场景 1 ===")
# result1 = agent_executor.invoke({
#         "input": "Python这个词有几个字母？"
#     })
# print(f"最终答案：{result1['output']}\n")

# template =(
#     "You are a helpful assistant that translates {input_language} to {output_language}"
# )
# system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)

# human_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_template)

# chat_prompt = ChatPromptTemplate.from_messages([
#     system_message_prompt,
#     human_message_prompt
# ])

# # 构建链
# chain = chat_prompt | llm | StrOutputParser()

# # 调用链（传递字典！）
# for chunk in chain.stream({
#     "input_language": "English",
#     "output_language": "French",  # 注意修正拼写错误
#     "text": "I love programming."
# }):
#     print(chunk, end="")

#print(msg)

# prompt_template = "Tell me a {language} {adjective} joke"
# prompt = PromptTemplate(
#     input_variables=["language","adjective"], template=prompt_template
# )

# chain = prompt | llm | StrOutputParser()

# msg = chain.invoke({"language": "France", "adjective": "funny"})
# print(msg)

# msg = chat_prompt.format_messages(
#     input_language = "English",
#     output_language = "Franch",
#     text = "I love programming."
# )
# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]
# llm.invoke(messages)


# for chunk in llm.stream(msg):
#     print(chunk.text(), end="")

# msg = llm.invoke([
#     HumanMessage(
#         content= (
#             "Translate this sentence from English to French."
#             "I love programming."
#         )
#     )
# ])

# print(msg.content)

# for chunk in llm.stream(messages):
#     print(chunk.text(), end="")

# from openai import OpenAI

# client = OpenAI(api_key="sk-e4e5862ddc5046f496cdb306775085cb", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)

# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model = "deepseek-r1:7b",
#     temperature = 0.8,
#     num_predict = 256,
#     # other params ...
# )

# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]
# llm.invoke(messages)

# messages = [
#     ("human", "Return the words Hello World!"),
# ]
# for chunk in llm.stream(messages):
#     print(chunk.text(), end="")