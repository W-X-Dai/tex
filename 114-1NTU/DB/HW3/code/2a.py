import duckdb
import matplotlib.pyplot as plt

con = duckdb.connect('my_database.db')

df = con.sql("""
-- list the hard questions
with hard_questions as(
    select question_id as id, count(*) as total_ans, sum(is_correct) as total_ac
    from answers
    group by question_id
    having total_ans >= 1000 and total_ac <= 500
),-- list the rank for all users
first_correct_pre as(
    select question_id, user_id, cost_time, created_at, rank() over(partition by question_id, user_id order by created_at asc) as rnk
    from answers
    where is_correct = 1
),-- first correct for each user
first_correct as(
    select question_id, user_id, cost_time, created_at
    from first_correct_pre
    where rnk = 1
),--rank users for hard questions
usr_rnk as(
    select question_id, user_id, rank() over(partition by question_id order by cost_time asc, created_at asc) as rnk
    from first_correct fc
    join hard_questions hq on hq.id = fc.question_id
)
select user_id, count(distinct question_id) as appr_cnt
from usr_rnk
where rnk <= 3
group by user_id
having appr_cnt >= 2
order by appr_cnt desc, user_id asc;
""").df()
print(df)

plt.figure(figsize=(12,6))
plt.bar(df["user_id"].astype(str), df["appr_cnt"])
plt.xlabel("User ID")
plt.ylabel("Number of Ranked Hard Questions")
plt.title("Leaderboard Top Users (Appearing in >= 2 Hard Questions)")
plt.tight_layout()
plt.savefig("2a_leaderboard.png")
plt.show()
