def get_current_date() -> str:
    """
    Returns the current date in YYYY-MM-DD format.
    """
    from datetime import date

    return date.today().strftime("%Y-%m-%d")
