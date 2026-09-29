"""An older database file missing a newer column is upgraded in place by init_db()."""
import sqlite3

from sqlalchemy import create_engine, inspect

from backend.database import db as dbmod


def test_missing_column_is_added(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE comparisons (id INTEGER PRIMARY KEY, document_id INTEGER, position INTEGER)")
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(dbmod, "engine", engine)
    dbmod.init_db()
    cols = {c["name"] for c in inspect(engine).get_columns("comparisons")}
    assert {"user_clause_label", "difference", "standard_value"} <= cols
    assert inspect(engine).has_table("users")
