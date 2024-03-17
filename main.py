import io

import discord
import os
from discord import app_commands
from dotenv import load_dotenv


from scrape import scrape_website, scrape_resume
from prompt import query_model

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


@client.event
async def on_ready():
    print("Enabling...")
    await client.change_presence(activity=discord.Game(name="Testing"))
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} commands")
    except Exception as e:
        print(e)


@tree.command(name="prepare", description="Prepare for meetings to build rapport with other industry professionals!")
@app_commands.describe(resume="Upload your own personal resume!")
@app_commands.describe(url="Upload the company or employer's website URL.")
async def hello_world(interaction, resume: discord.Attachment, url: str):
    await interaction.response.defer()
    pdf_data = await resume.read()
    # Create a BytesIO object from the PDF data
    pdf_buffer = io.BytesIO(pdf_data)

    website_data = scrape_website(url)
    resume_data = scrape_resume(pdf_buffer)

    statement = query_model(website_data, resume_data)

    max_length = 1999
    if len(statement) <= max_length:
        await interaction.followup.send(statement)
    else:
        truncated_content = statement[:max_length]
        await interaction.followup.send(truncated_content)

client.run(DISCORD_TOKEN)
