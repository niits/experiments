from fastmcp import FastMCP
from datetime import datetime, timedelta

mcp = FastMCP(
    name="Calculation Service",
    description="This service is used to perform calculations, such as date/datetime subtraction and addition.",
    version="0.1.0",
    instructions="To use this service, please provide parameters in the correct format.",
)


@mcp.tool()
def diff_datetime(
    date1: datetime,
    date2: datetime,
) -> str:
    # Return in format days, hours, minutes, seconds for example 2023-10-01 12:00:00 - 2023-10-01 11:00:00 = 1 day, 1 hour, 0 minutes, 0 seconds

    diff = date1 - date2

    return f"{diff.days} days, {diff.seconds // 3600} hours, {(diff.seconds // 60) % 60} minutes, {diff.seconds % 60} seconds"


@mcp.tool()
def add_datetime(
    date: datetime,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> str:
    # Return in format YYYY-MM-DD HH:MM:SS
    date = date + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return date.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def get_current_time() -> str:
    # Return in format YYYY-MM-DD HH:MM:SS
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

http_app = mcp.http_app(transport="sse")
