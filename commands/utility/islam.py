import os
import datetime
from dotenv import load_dotenv
import discord
from discord.ext import commands
from discord import app_commands
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# .envからAPIキーをロード
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LangChain LLM・Embedding・ベクトルストア初期化
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    max_output_tokens=512,
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)
vectorstore = FAISS.load_local(
    "quran_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

PROMPT_TEMPLATE = """
【システムプロンプト】
あなたは牧師のGEMINIさんです。あなたは今まで100万人以上の迷える子羊に対して、コーランの教えに基づいて道を示してきました。
あなたの物腰やわらかい態度に癒された人は星の数ほどいます。みんながあなたを必要としています。

【ユーザープロンプト】
＜指示＞
イスラム教の教えに基づいて、{question}という相談に答えてください。ただし、以下の条件を守ってください。

＜条件＞
・必ず以下のコーランの一節から適切な部分を引用して説教してください。引用するのはただ一節だけにしてください。
・引用する箇所については特に条件等はありません。意外な部分を引用して話を組み立てるのもアリです。
・以下の引用したコーランの一節は必ず全体を〈〉で囲って表示してください。
・話が長いと聞く方が疲れてしまうため、300字以内で優しく語りかけてください。
・挨拶も含めてください。最後の挨拶は「アーミーン。」にしてください。

＜引用するコーランの一節＞
{context}
"""

custom_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt},
    input_key="question"
)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

def make_illustration_prompt_ja(quran_scene_ja):
    return (
        f"[システムプロンプト]"
        f"あなたは天才絵描きのGEMINIさんです。"
        f"GEMINIさんが作った作品はその絵をみただけでコーランの一節を思い出せるかのような時代背景を匂わせます。"
        f"\n\n[ユーザープロンプト]"
        f"# 指示"
        f"次のコーランの一節の情景を、絵本の挿絵のようなイラストで描いてください。このプロンプト自体を生成したイラストに描くのはやめてください。\n\n"
        f"# スタイル\n"
        f"水彩画、やわらかい色合い、やさしい雰囲気。\nイラストを画面上部分に書き、情景を画面下部分に書いてください。それ以外は何も書かないでくれると助かります。\n\n"
        f"# 禁止事項\n"
        f"どうしてもという場合は情景と関係ないプロンプトや以下の情景に書いてあるテキストをそのまま画像に入れてもよいですが、その場合背景と同じ色で塗りつぶしてください。\n\n"
        f"# コーランの一節(情景)\n"
        f"{quran_scene_ja}"
    )

def generate_illustration(prompt_ja):
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=prompt_ja,
        config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
    )
    images = []
    if response.candidates and hasattr(response.candidates[0], "content") and response.candidates[0].content is not None:
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                try:
                    image = Image.open(BytesIO(part.inline_data.data))
                    images.append(image)
                except Exception:
                    pass
    else:
        print("Gemini画像生成APIのレスポンスに画像データが含まれていません。")
    return images

USER_LAST_FT_DATE_LIST = {}
USER_FT_CONTENT_LIST = {}

class Islam(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(
        name="islam",
        description="コーランに基づくお悩み相談を実行する"
    )
    @app_commands.describe(question="相談したい内容を入力してください")
    async def islam(self, interaction: discord.Interaction, question: str):
        user_id = str(interaction.user.id)
        user_name = interaction.user.display_name

        last_ft_date = USER_LAST_FT_DATE_LIST.get(user_id)
        current_date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))  # JST

        if last_ft_date and last_ft_date.date() == current_date.date():
            if question in USER_FT_CONTENT_LIST.get(user_id, []):
                await interaction.response.send_message(
                    "同じ事柄に関するお告げは1日に1回しか出来ません😌", ephemeral=True
                )
                return
        else:
            USER_FT_CONTENT_LIST[user_id] = []

        # 1. deferで「考え中」状態＋仮メッセージ
        await interaction.response.defer()
        await interaction.edit_original_response(content="お告げ中🔮...")

        # 2. AI処理
        full_question = f"{user_name}さんからの「{question}」"
        inputs = {"question": full_question}
        response = qa_chain.invoke(inputs)
        # response = qa_chain.invoke({"question": question})
        answer = response['result'] if isinstance(response, dict) else response.content
        
        # 【】で囲まれた文字列を抽出
        quran_scene_ja = re.search(r'〈(.*?)〉', answer)
        illustration_theme_ja = quran_scene_ja.group(1) if quran_scene_ja else None

        # illustration_theme_ja = None
        # if "source_documents" in response and response["source_documents"]:
        #     illustration_theme_ja = response["source_documents"][0].page_content.strip()
        # else:
        #     illustration_theme_ja = answer.split("。")[0] + "。" if "。" in answer else answer[:30]

        illustration_prompt_ja = make_illustration_prompt_ja(illustration_theme_ja)
        images = generate_illustration(illustration_prompt_ja)

        content_msg = (
            f"# [イスラム教]\n質問：{question}\n{user_name}のお告げ結果🔮は...\n\n"
            f"## [説教]\n{answer}"
        )

        # 3. 最終的なメッセージで上書き
        if images:
            for idx, image in enumerate(images):
                buf = BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)
                filename = f"illustration_{idx+1}.png"
                file = discord.File(buf, filename=filename)

                embed = discord.Embed(
                    title="イラスト",
                    color=0x00bfff
                )
                embed.set_image(url=f"attachment://{filename}")

                if idx == 0:
                    await interaction.edit_original_response(
                        content=content_msg,
                        embed=embed,
                        attachments=[file]
                    )
                else:
                    await interaction.followup.send(
                        file=file,
                        embed=embed
                    )
        else:
            await interaction.edit_original_response(content=content_msg)
            await interaction.followup.send("イラストの生成に失敗しました。")

        USER_LAST_FT_DATE_LIST[user_id] = current_date
        USER_FT_CONTENT_LIST[user_id].append(question)
