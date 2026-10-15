"""启动本地 PostgreSQL(pgvector)，等价替代 docker-compose 中的 pgvector/pgvector:pg16。

用法:
  uv run --no-sync python scripts/local_pg.py

依赖 pgserver 包（自带 PG+pgvector 二进制），一次性安装:
  uv pip install --default-index https://pypi.tuna.tsinghua.edu.cn/simple pgserver

注意：
- 不走 pgserver.get_server()——它启动时固定加 `-h ""`（仅 unix socket，无 TCP），
  应用的 PG_DSN（localhost:5432）会连不上。这里直接用 pgserver 自带的 pg_ctl 以
  `listen_addresses=localhost` 启动（写在 .pgdata/postgresql.conf 末尾）。
- 幂等；脚本退出后 PostgreSQL 保持运行。停止：pg_ctl -D .pgdata stop。
- 数据目录 .pgdata/ 已 gitignore，勿提交。
"""

import sys
from pathlib import Path

import pgserver
import psycopg

ROOT = Path(__file__).resolve().parent.parent
PGDATA = ROOT / ".pgdata"
DSN = "postgresql://invoice:invoice@localhost:5432/invoice_agent"
ROLE, PASSWORD, DB = "invoice", "invoice", "invoice_agent"


def main() -> int:
    # 1. 初始化数据目录（首次）
    if not (PGDATA / "PG_VERSION").exists():
        pgserver.initdb(
            ["--auth=trust", "--auth-local=trust", "--encoding=utf8", "-U", "postgres"],
            pgdata=PGDATA,
        )
        with open(PGDATA / "postgresql.conf", "a") as f:
            f.write("\nlisten_addresses = 'localhost'\n")
        print(f"[local_pg] 数据目录已初始化: {PGDATA}")

    # 2. 已运行则跳过；否则以 TCP 模式启动
    ready = ""
    if (PGDATA / "postmaster.pid").exists():
        try:
            ready = pgserver.pg_isready(["-h", "localhost", "-p", "5432"])
        except Exception as err:  # noqa: BLE001 — 探测失败按未启动处理
            print(f"[local_pg] pg_isready 探测失败，按未启动处理: {err}")
    if "accepting connections" in ready:
        print("[local_pg] PostgreSQL 已在运行 (localhost:5432)")
    else:
        pgserver.pg_ctl(
            ["-w", "-o", f'-k "{PGDATA}"', "-l", str(PGDATA / "logfile"), "start"],
            pgdata=PGDATA,
        )
        print("[local_pg] PostgreSQL 已启动 (TCP localhost:5432)")

    # 3. 幂等创建角色 / 数据库 / pgvector 扩展
    with psycopg.connect(host="localhost", port=5432, user="postgres",
                         dbname="postgres", autocommit=True) as conn:
        roles = {r[0] for r in conn.execute("SELECT rolname FROM pg_roles").fetchall()}
        if ROLE not in roles:
            conn.execute(f"CREATE ROLE {ROLE} LOGIN PASSWORD '{PASSWORD}'")
            print(f"[local_pg] 已创建角色 {ROLE}")
        dbs = {r[0] for r in conn.execute("SELECT datname FROM pg_database").fetchall()}
        if DB not in dbs:
            conn.execute(f"CREATE DATABASE {DB} OWNER {ROLE}")
            print(f"[local_pg] 已创建数据库 {DB}")

    with psycopg.connect(host="localhost", port=5432, user="postgres",
                         dbname=DB, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("[local_pg] vector 扩展就绪")

    # 4. 用应用同款 DSN 验证
    with psycopg.connect(DSN) as conn:
        print("[local_pg] 连接验证:",
              conn.execute("SELECT current_user, current_database()").fetchone())
    print("[local_pg] DSN:", DSN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
