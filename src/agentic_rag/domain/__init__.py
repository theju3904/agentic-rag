"""Framework-free domain entities shared by every layer of the application.

Nothing in this package imports SQLAlchemy, Pydantic, FastAPI, or any vendor
SDK. Storage, API, and pipeline layers translate to/from these dataclasses so
that the core business types never leak infrastructure concerns.
"""
