from writing_tool.logging_config import get_logger

def main():
    logger = get_logger(__name__)
    logger.info("Hello from writing-tool!")


if __name__ == "__main__":
    main()
