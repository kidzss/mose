import pymysql
import csv


def import_csv_to_table(database, table_name, csv_file_path, columns, host='localhost', user='root', password=''):
    """
    将 CSV 文件导入到 MySQL 数据库的指定表中。

    :param database: 数据库名称
    :param table_name: 目标表名称
    :param csv_file_path: CSV 文件路径
    :param columns: 表中的列名列表，顺序需与 CSV 文件一致
    :param host: 数据库主机地址，默认 localhost
    :param user: 数据库用户名，默认 root
    :param password: 数据库密码，默认空
    """
    # 数据库连接
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        charset='utf8mb4'
    )

    try:
        with connection.cursor() as cursor:
            # 打开 CSV 文件
            with open(csv_file_path, 'r') as file:
                reader = csv.DictReader(file)

                # 遍历每一行，插入数据
                for row in reader:
                    placeholders = ', '.join(['%s'] * len(columns))
                    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    values = tuple(row[col] for col in columns)
                    cursor.execute(sql, values)

            # 提交事务
            connection.commit()
            print(f"数据成功导入到表 '{table_name}'")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        connection.close()


if __name__ == '__main__':
    # 导入 nasdaq100_stocks.csv 到 uss_nasdaq_stocks 表
    # import_csv_to_table(
    #     database='mose',
    #     table_name='uss_nasdaq_stocks',
    #     csv_file_path='nasdaq100_stocks.csv',
    #     columns=['Code', 'Name', 'Sector', 'Industry']
    # )

    # # 导入 sp500_stocks.csv 到 uss_sp_stocks 表
    # import_csv_to_table(
    #     database='mose',
    #     table_name='uss_sp_stocks',
    #     csv_file_path='sp500_stocks.csv',
    #     columns=['Code', 'Name', 'Sector', 'Industry']
    # )

    # 导入 uss_etf_stocks.csv 到 uss_etf_stocks 表
    import_csv_to_table(
        database='mose',
        table_name='uss_etf_stocks',
        csv_file_path='uss_etf_stocks.csv',
        columns=['Code', 'Name', 'Sector', 'Industry']
    )
