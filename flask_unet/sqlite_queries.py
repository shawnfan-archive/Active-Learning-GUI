import sqlite3
import pandas as pd

def query_db(db_path,
             log_name,
             target_col='*',
             condition_col=None,
             condition_val=None,
             output_type=None):
    """
    Arguments: 
        db_path: a database's file path
        log_name: the name of the log from which values are selected
        target_col: the name of the column from which values are selected
        condition_col: the column name of the filter
        condition_val: the value(s) of the filter
        output_type: the type of the returned value(s)
    Returns:
        output: 
    """

    conn = sqlite3.connect(db_path)

    query = " SELECT " + target_col + " from " + log_name
    condition = ""
    if condition_col != None:
        condition = " where "
        for i in range(len(condition_col)):
            if i > 0:
                condition += " and "
            condition = condition + condition_col[i]
            if type(condition_val[i]) == list:
                # !- if condition_val has only 1 element, converting it tuple introduces a comma that causes a formatting error in the sqlite query
                if len(condition_val[i]) == 1:
                    condition = condition + "=" + str(condition_val[i][0])
                else: 
                    condition = condition + " in " + str(tuple(condition_val[i]))
            else:
                condition = condition + " = " + str(condition_val[i])

    query = query + condition

    df = pd.read_sql_query(query, conn)

    conn.close()

    if output_type == str:
        output = df[target_col].to_string(index=False)
        output = output.replace(' ', '')  #Remove space
    elif output_type == list:
        output = df[target_col].to_list()
    else:
        output = df

    return output

def delete_from_db(db_path, log_name, condition_col, condition_val):
    """
    Delete entries from a table in a database under a SINGLE condition

    Arguments: 
        db_path: a database's file path
        log_name: the name of the log from which entries are deleted
        condition_col: the column name of the filter
        condition_val: the value(s) of the filter

    Returns: None
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if type(condition_val) == list:
        cur.execute(" DELETE FROM " + log_name + " WHERE " + condition_col + " IN " + str(tuple(condition_val)))
    else: 
        cur.execute(" DELETE FROM " + log_name + " WHERE " + condition_col + " = " +  str(condition_val))

    conn.commit()
    conn.close()

    return "Entry removed!"

def insert_into_db(db_path, log_name, col_names, entry):
    """
    Insert an entry/entries into a database

    Arguments:
        db_path: a database's file path 
        log_name: the name of a log into which an entry is inserted 
        col_names: an array of names of all columns in the table except the column with the primary key
        entry: a tuple that contains values to be inserted
    
    Returns:
        entry_id: the primary key (usually the entry id) of the inserted entry
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(" INSERT INTO " + log_name + " " + str(col_names) +
                " VALUES " + str(entry))
    entry_id = cur.lastrowid

    conn.commit()
    conn.close()

    return entry_id

def update_db(db_path, log_name, target_col, target_val, condition_col, condition_val):
    """
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(" UPDATE " + log_name + " SET " + target_col + " = " + str(target_val) +
     " where " + condition_col + " = " + str(condition_val))

    conn.commit()
    conn.close()
    
    return None