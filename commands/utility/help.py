import discord
from discord import app_commands
from discord.ext import commands

class Tarot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(
        name="help",
        description="牧師botの説明"
    )
    async def help(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="牧師bot",
            description="あなたの悩みに対して、牧師による説教が受けられるdiscord botです🐦"
            # url="https://atelier365.net/blog/tarot-spread/"
        )
        embed.set_thumbnail(url="https://static.vecteezy.com/system/resources/previews/040/970/171/non_2x/ai-generated-robot-priest-in-church-modern-world-artificial-intelligence-replacement-concept-robot-as-spiritual-leader-free-photo.jpg")

        content = (
            "# 牧師bot🔮へようこそ！\n"
            "このbotは牧師があなたのお悩みに対して説教をしてくれます！\n\n"
            "## 説教コマンドの種類：\n"
            "**/christianity** … 聖書に基づいて、あなたのお悩み相談をします[学業、金運、恋愛etc…]。\n\n"
            "## お告げのコツ🐦\n"
            "明確な答えを得るには、自分が知りたいことをはっきりさせてお願いすることが大切です。"
            "曖昧なまま相談してしまうと、思ったのと違う内容について説教されてしまうかもしれません。"
            "まず「**自分は何が困っているのか**」をはっきり言語化することで良い質問に近づくことができるでしょう。\n"
            "ex.) 友人と喧嘩してしまいました。仲直りしたいです。\n\n"
            "参考文献：（編集中）"
        )

        await interaction.response.send_message(
            content=content,
            embed=embed,
            ephemeral=False  # 必要に応じてTrueに変更
        )

