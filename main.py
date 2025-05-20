import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import importlib.util
import pathlib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import threading
from commands.utility.christianity import Christianity

# .envからトークン等をロード
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
PORT = int(os.getenv("PORT", 3000))
HOST = "0.0.0.0" if "RENDER" in os.environ else "localhost"
GUILDID = os.getenv("DISCORD_GUILDID")

# Discord Botのセットアップ
intents = discord.Intents.default()
# intents.message_content = True  # メッセージ内容の読み取り権限を無効化
bot = commands.Bot(command_prefix="!", intents=intents)

# コマンド自動ロード
async def load_cogs():
    commands_dir = pathlib.Path(__file__).parent / "commands"
    for utility_dir in commands_dir.glob("*utility"):
        for py_file in utility_dir.glob("*.py"):
            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # コグとして登録（コグクラス名はファイル内で定義されている必要あり）
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, commands.Cog):
                    await bot.add_cog(obj(bot))

@bot.event
async def on_ready():
    # グローバルのスラッシュコマンド登録処理
    await load_cogs()  # コグの非同期ロード
    await bot.tree.sync() # スラッシュコマンドの自動同期

    # ギルドのスラッシュコマンド登録処理
    guild = discord.Object(id=GUILDID)  # あなたのサーバーID
    await bot.tree.sync(guild=guild)
    print("ギルドコマンドを同期しました")
    
    print(f"準備完了！{bot.user}としてログインしました！")

# FastAPIによるヘルスチェックサーバー
app = FastAPI()

@app.get("/ping")
async def ping():
    return HTMLResponse("""
        <!DOCTYPE html>
        <html lang="ja">
            <head>
                <title>Document</title>
            </head>
            <body>
                <p>Ping!</p>
            </body>
        </html>
    """)

def run_fastapi():
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

# FastAPIサーバーを別スレッドで起動
threading.Thread(target=run_fastapi, daemon=True).start()

# スラッシュコマンドのエラーハンドリング
@bot.tree.error
async def on_app_command_error(interaction, error):
    print(error)
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message("このコマンドの実行中にエラーが発生しました！", ephemeral=True)
        else:
            await interaction.followup.send("このコマンドの実行中にエラーが発生しました！", ephemeral=True)
    except Exception as e:
        print(f"エラーハンドラーでエラーが発生: {e}")

# Bot起動
bot.add_cog(Christianity(bot))
bot.run(TOKEN)