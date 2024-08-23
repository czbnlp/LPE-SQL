import sqlite3
import signal
from contextlib import contextmanager

# 使用信号量来实现超时中断
@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Function call timed out")

    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    except TimeoutError as e:
        raise e
    finally:
        signal.alarm(0)  # 取消闹钟信号

def retry_on_timeout(seconds=5, max_retries=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = max_retries + 1  # 包括首次尝试在内的最大尝试次数
            while retries > 0:
                retries -= 1
                try:
                    with timeout(seconds):
                        return func(*args, **kwargs)
                except TimeoutError:
                    if retries == 0:
                        raise  # 如果没有剩余重试机会，抛出异常
                    else:
                        continue  # 重试

        return wrapper
    return decorator



def execute_sql(predicted_sql, db_path):
    # print(f"sql: {predicted_sql}")
    conn = None
    ans = ""
    conn = sqlite3.connect(db_path)
    error_information = None
    with conn:
        try:
            cur = conn.cursor()
            cur.execute(predicted_sql)
            ans = cur.fetchall()
        except sqlite3.Error as e:
            # print(db_path)
            error_information = str(e)
    ans = ['\n'.join(map(str, item)) for item in ans]
    ans = '\n'.join(ans)
    return ans, error_information