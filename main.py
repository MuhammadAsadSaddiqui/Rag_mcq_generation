import asyncio
from worker import start_worker

async def main():
    print("Starting MCQ Worker...")
    await start_worker()

if __name__ == "__main__":
    asyncio.run(main())