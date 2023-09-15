############
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#dataframe okundu
df = pd.read_excel("no_nans_data.xlsx")
df = df.copy()
df.head()
df.columns
df.shape

#veri temizleme
keyword_1 = "(17/18)"
columns_to_drop = [col for col in df.columns if keyword_1 in col]
df.drop(columns= columns_to_drop, inplace=True)

keyword_2 = "(18/19)"
columns_to_drop = [col for col in df.columns if keyword_2 in col]
df.drop(columns= columns_to_drop, inplace=True)

keyword_3 = "(20/21)"
columns_to_drop = [col for col in df.columns if keyword_3 in col]
df.drop(columns= columns_to_drop, inplace=True)



#kategorik ve numerik değişkenler
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]


for col in df.columns:
    if col in num_cols:
        print(df.groupby("Position")[col].mean())



df.head()




#DEFENDER SEGMENTATİON
######################
df_defender = pd.DataFrame()
df_defender = df[df["Position"] == "Defender"]


defender_columns = [col for col in df.columns if col in ["Position","Age","Player","Nation","MP (19/20)",\
"Min (19/20)", "Goals/Shots (19/20)","Passes Leading to Shot Attempt (19/20)","Defensive Actions Leading to Shot Attempt (19/20)", \
"Touches in Defensive Penalty Box (19/20)", "Touches in Defensive 3rd (19/20)","Touches in Midfield 3rd (19/20)","Touches in Open-play (19/20)",\
"Total Carries (19/20)" ,"Total Distance Carried the Ball (19/20)","% of Times Successfully Received Pass (19/20)" ,"Pass Completion % (All pass-types) (19/20)",\
"Total Tackles Won (19/20)","Total Defensive Blocks (19/20)","Total Shots Blocked (19/20)","Goal Saving Blocks (19/20)", \
"Times blocked a Pass (19/20)", "Aerial Duel Won (19/20)","Aerial Duel Lost (19/20)","Total Loose Balls Recovered (19/20)"]]

df_defender = df_defender[defender_columns]

df_defender.head()

##################
#Total Score Oluşturulması
##################
df_defender["Aerial_Duel_Total"] = df_defender["Aerial Duel Won (19/20)"] - df_defender["Aerial Duel Lost (19/20)"]


df_defender["Age_Score"] = pd.qcut(df_defender["Age"], 5, labels= [5,4,3,2,1])
df_defender["MP_Score"] = pd.qcut(df_defender["MP (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Min_Score"] = pd.qcut(df_defender["Min (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Passes_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_defender["Passes Leading to Shot Attempt (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Defensive_Actions_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_defender["Defensive Actions Leading to Shot Attempt (19/20)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_defender["Touches_in_Defensive_Penalty_Box_Score"] = pd.qcut(df_defender["Touches in Defensive Penalty Box (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Touches_in_Defensive_3rd_Score"] = pd.qcut(df_defender["Touches in Defensive 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Total_Disctance_Score"] = pd.qcut(df_defender["Total Distance Carried the Ball (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Pass_Completion_Score"] = pd.qcut(df_defender["Pass Completion % (All pass-types) (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["%_of_Times_Successfully_Received_Pass_Score"] = pd.qcut(df_defender["% of Times Successfully Received Pass (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["total_Loose_Balls_Recovered_Score"] = pd.qcut(df_defender["Total Loose Balls Recovered (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Total_Tackles_Won_Score"] = pd.qcut(df_defender["Total Tackles Won (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Total_Defensive_Blocks_Score"] = pd.qcut(df_defender["Total Defensive Blocks (19/20)"], 5, labels= [1,2,3,4,5])
df_defender["Aerial_Duel_Total_Score"] = pd.qcut(df_defender["Aerial_Duel_Total"], 5, labels= [1,2,3,4,5])


df_defender["Total_Score"] = (df_defender["MP_Score"].astype(int) +  df_defender["Min_Score"].astype(int) + \
                              df_defender["Passes_Leading_to_Shot_Attempt_Score"].astype(int) + df_defender["Defensive_Actions_Leading_to_Shot_Attempt_Score"].astype(int) + \
                              df_defender["Touches_in_Defensive_Penalty_Box_Score"].astype(int) + df_defender["Touches_in_Defensive_3rd_Score"].astype(int) + \
                              df_defender["%_of_Times_Successfully_Received_Pass_Score"].astype(int) + df_defender["Total_Tackles_Won_Score"].astype(int) + \
                              df_defender["Total_Defensive_Blocks_Score"].astype(int) + df_defender["Aerial_Duel_Total_Score"].astype(int) +\
                              df_defender["Total_Disctance_Score"].astype(int) + df_defender["Pass_Completion_Score"].astype(int) + df_defender["total_Loose_Balls_Recovered_Score"].astype(int))
df_defender["Total_Score"].sort_values(ascending = False)

df_defender["Total_Score"].describe().T
df_defender[df_defender["Total_Score"] > 57]

df_defender["Total_Score"].quantile(0.90)

##################
#Segmentleme
##################
df_defender['Performance'] = pd.cut(x=df_defender['Total_Score'], bins=[13, 24, 39, 49, 57, 63],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])


df_defender[df_defender['Performance'] == "High_performance"]
df_defender[df_defender['Age'] < 22]

df_defender["Age"].describe().T

df_defender['Age_Cat'] = pd.cut(x=df_defender['Age'], bins=[17, 22, 27, 32, 37],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])


df_defender["Segment19_20"] = df_defender["Age_Cat"].astype(str) + "_" + df_defender["Performance"].astype(str)




df_defender.loc[(df_defender["Performance"] == "Under_expected"), "Performance_Score_19/20"] = "1"
df_defender.loc[(df_defender["Performance"] == "Open_to_development"), "Performance_Score_19/20"] = "2"
df_defender.loc[(df_defender["Performance"] == "Player_with_high_potential"), "Performance_Score_19/20"] = "3"
df_defender.loc[(df_defender["Performance"] == "High_performance"), "Performance_Score_19/20"] = "4"
df_defender.loc[(df_defender["Performance"] == "Flawless"), "Performance_Score_19/20"] = "5"

df_defender.head()

#final dataframe

final_defender_19_20 = pd.DataFrame()
final_columns_defender= [col for col in df_defender.columns if col in ["Player", "Segment19_20","Performance_Score_19/20"]]

final_defender_19_20 = df_defender[final_columns_defender]












df.head()



#######################
#Midfield Segmantation
#######################

df_midfield = pd.DataFrame()
df_midfield =df[df["Position"] == "midfield"]
df_midfield.head()




midfield_columns = [col for col in df.columns if col in ["Player","Age","Position","Total Distance of Completed Progressive Passes (All Pass-types) (19/20)",\
"Ast (19/20)", "Gls (19/20)", "Non-Penalty Goals (19/20)","Min (19/20)","Shots on Target% (19/20)", "Goals/Shots on Target (19/20)","Passes Leading to Shot Attempt (19/20)",\
"Dribbles Leading to Shot Attempt (19/20)","Defensive Actions Leading to Shot Attempt (19/20)" ,"Passes Leading to Goals (19/20)", \
"Touches in Midfield 3rd (19/20)", "Touches in Attacking 3rd (19/20)","Touches in Defensive 3rd (19/20)", "Touches in Open-play (19/20)",\
"Total Distance Carried the Ball (19/20)","Total Distance Carried the Ball in Forward Direction","Number of Times Player was Pass Target (19/20)" ,\
"% of Times Successfully Received Pass (19/20)"  ,"Pass Completion % (All pass-types) (19/20)","Completed passes that enter Final 3rd (19/20)",\
"Total Tackles Won (19/20)","Total Players Tackled + Total Interceptions (19/20)","MP (19/20)","Club","Nation","League"]]

df_midfield = df_midfield[midfield_columns]

df_midfield["Age_Score"] = pd.qcut(df_midfield["Age"], 5, labels= [5,4,3,2,1])
df_midfield["MP_Score"] = pd.qcut(df_midfield["MP (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Min_Score"] = pd.qcut(df_midfield["Min (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Ast_Score"] = pd.qcut(df_midfield["Ast (19/20)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_midfield["Gls_Score"] = pd.qcut(df_midfield["Gls (19/20)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_midfield["Non-Penalty_goals_Score"] = pd.qcut(df_midfield["Non-Penalty Goals (19/20)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_midfield["Goals/Shots_on_Target_Score"] = pd.qcut(df_midfield["Goals/Shots on Target (19/20)"].rank(method = "first"), 5, labels= [1,2,3,4,5])
df_midfield["PassesLeadingtoShotAttempt_Score"] = pd.qcut(df_midfield["Passes Leading to Shot Attempt (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Defensive_Actions_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_midfield["Defensive Actions Leading to Shot Attempt (19/20)"].rank(method = "first"), 5, labels= [1,2,3,4,5])
df_midfield["TouchesinDefensive3rd_score"] = pd.qcut(df_midfield["Touches in Defensive 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Touches in Midfield 3rd_Score"] = pd.qcut(df_midfield["Touches in Midfield 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Touches in Attacking 3rd_Score"] = pd.qcut(df_midfield["Touches in Attacking 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Touches in Open-play_Score"] = pd.qcut(df_midfield["Touches in Open-play (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Number of Times Player was Pass Target_Score"] = pd.qcut(df_midfield["Number of Times Player was Pass Target (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Pass Completion % (All pass-types)_Score"] = pd.qcut(df_midfield["Pass Completion % (All pass-types) (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Completed passes that enter Final 3rd_Score"] = pd.qcut(df_midfield["Completed passes that enter Final 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_midfield["Total Tackles Won_Score"] = pd.qcut(df_midfield["Total Tackles Won (19/20)"], 5, labels= [1,2,3,4,5])


df_midfield["Total_Score"] = (df_midfield["MP_Score"].astype(int) +  df_midfield["Min_Score"].astype(int) + \
                              df_midfield["Ast_Score"].astype(int) + df_midfield["Gls_Score"].astype(int) + \
                              df_midfield["Non-Penalty_goals_Score"].astype(int) + df_midfield["Goals/Shots_on_Target_Score"].astype(int) + \
                              df_midfield["PassesLeadingtoShotAttempt_Score"].astype(int) + df_midfield["Defensive_Actions_Leading_to_Shot_Attempt_Score"].astype(int)+ \
                              df_midfield["TouchesinDefensive3rd_score"].astype(int) + df_midfield["Touches in Midfield 3rd_Score"].astype(int) + \
                              df_midfield["Touches in Attacking 3rd_Score"].astype(int) + df_midfield["Touches in Open-play_Score"].astype(int) + df_midfield["Number of Times Player was Pass Target_Score"].astype(int)+ \
                              df_midfield["Pass Completion % (All pass-types)_Score"].astype(int) + df_midfield["Completed passes that enter Final 3rd_Score"].astype(int)+ \
                              df_midfield["Total Tackles Won_Score"].astype(int))
df_midfield["Total_Score"].sort_values(ascending = False)



df_midfield["Total_Score"].describe().T
df_midfield["Total_Score"].quantile(0.90)


df_midfield['Performance'] = pd.cut(x=df_midfield['Total_Score'], bins=[15,32,45,56,67,78],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])


df_midfield.loc[(df_midfield["Performance"] == "Under_expected"), "Performance_Score_19/20"] = "1"
df_midfield.loc[(df_midfield["Performance"] == "Open_to_development"), "Performance_Score_19/20"] = "2"
df_midfield.loc[(df_midfield["Performance"] == "Player_with_high_potential"), "Performance_Score_19/20"] = "3"
df_midfield.loc[(df_midfield["Performance"] == "High_performance"), "Performance_Score_19/20"] = "4"
df_midfield.loc[(df_midfield["Performance"] == "Flawless"), "Performance_Score_19/20"] = "5"


df_midfield['Age_Cat'] = pd.cut(x=df_midfield['Age'], bins=[17, 22, 27, 32, 37],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])

df_midfield["Segment19_20"] = df_midfield["Age_Cat"].astype(str) + "_" + df_midfield["Performance"].astype(str)



df_midfield[df_midfield['Performance'] == "High_performance"].head()

df_midfield.head()


#final dataframe for midfield

final_midfield_19_20= pd.DataFrame()
final_columns_midfield= [col for col in df_midfield.columns if col in ["Player", "Segment19_20","Performance_Score_19/20"]]

final_midfield_19_20 = df_midfield[final_columns_midfield]

final_midfield["Segment19_20"].value_counts()

final










#######################
#Attack Segmantation
#######################



df_attack = pd.DataFrame()
df_attack =df[df["Position"] == "attack"]
df_attack.head()
df_attack.shape



attack_columns = [col for col in df.columns if col in ["Age", "MP (19/20)","Player","Club","Position","Nation","League","Min (19/20)", \
                                                   "Gls (19/20)","Ast (19/20)","Non-Penalty Goals (19/20)","Shots on Target% (19/20)",\
                                                   "Goals/Shots (19/20)","Goals Scored minus xG (19/20)","Passes Leading to Shot Attempt (19/20)",\
                                                   "Dribbles Leading to Shot Attempt (19/20)","Goal Creating Actions (19/20)","Passes Leading to Goals (19/20)",\
                                                   "Dribbles Leading to Goals (19/20)","Touches in Attacking 3rd (19/20)","Touches in Attacking Penalty Box (19/20)",\
                                                   "Total Successful Dribbles (19/20)","Carries into Attacking Penalty Box (19/20)","Total Failed Attempts at Controlling Ball (19/20)", \
                                                   "% of Times Successfully Received Pass (19/20)","Progressive Passes Received (19/20)","Completed passes that enter Penalty Box (19/20)",\
                                                   "Aerial Duel Won (19/20)","Aerial Duel Lost (19/20)","Tackles in Attacking 3rd (19/20)","Successful Pressure % (19/20)"]]


df_attack = df_attack[attack_columns]

df_attack.head()


df_attack["Aerial_Duel_Total"] = df_attack["Aerial Duel Won (19/20)"] - df_attack["Aerial Duel Lost (19/20)"]


df_attack["Age_Score"] = pd.qcut(df_attack["Age"], 5, labels= [5,4,3,2,1])
df_attack["MP_Score"] = pd.qcut(df_attack["MP (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Min_Score"] = pd.qcut(df_attack["Min (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Ast_Score"] = pd.qcut(df_attack["Ast (19/20)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_attack["Gls_Score"] = pd.qcut(df_attack["Gls (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Non-Penalty_goals_Score"] = pd.qcut(df_attack["Non-Penalty Goals (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Shots_on_Target%_Score"] = pd.qcut(df_attack["Shots on Target% (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Goals/Shots_Score"] = pd.qcut(df_attack["Goals/Shots (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Goal_scored_minus_xG_Score"] = pd.qcut(df_attack["Goals Scored minus xG (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Passes_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_attack["Passes Leading to Shot Attempt (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Dribbles_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_attack["Dribbles Leading to Shot Attempt (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Goal_Creating_Actions_Score"] = pd.qcut(df_attack["Goal Creating Actions (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Passes_Leading_to_Goals_Score"] = pd.qcut(df_attack["Passes Leading to Goals (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Dribbles_Leading_to_Goals_Score"] = pd.qcut(df_attack["Dribbles Leading to Goals (19/20)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_attack["Touches_in_Attacking_3rd_Score"] = pd.qcut(df_attack["Touches in Attacking 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Touches_in_Attacking_Penalty_Box_Score"] = pd.qcut(df_attack["Touches in Attacking Penalty Box (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Carries_into_Attacking_Penalty_Box_Score"] = pd.qcut(df_attack["Carries into Attacking Penalty Box (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Total_Successful_Dribbles_Score"] = pd.qcut(df_attack["Total Successful Dribbles (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Total_Failed_Attempts_at_Controlling_Ball_Score"] = pd.qcut(df_attack["Total Failed Attempts at Controlling Ball (19/20)"].rank(method="first"), 5, labels= [5,4,3,2,1])
df_attack["%Progressive_Passes_Received_Score"] = pd.qcut(df_attack["Progressive Passes Received (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["%Completed_passes_that_enter_Penalty_Box_Score"] = pd.qcut(df_attack["Completed passes that enter Penalty Box (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["%Aerial_Duel_Total_Score"] = pd.qcut(df_attack["Aerial_Duel_Total"], 5, labels= [1,2,3,4,5])
df_attack["%_of_Times_Successfully_Received_Pass_Score"] = pd.qcut(df_attack["% of Times Successfully Received Pass (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Tackles_in_Attacking_3rd_Score"] = pd.qcut(df_attack["Tackles in Attacking 3rd (19/20)"], 5, labels= [1,2,3,4,5])
df_attack["Successful_Pressure_%_Score"] = pd.qcut(df_attack["Successful Pressure % (19/20)"], 5, labels= [1,2,3,4,5])




df_attack["Total_Score"] = (df_attack["MP_Score"].astype(int) +  df_attack["Min_Score"].astype(int) + \
                              df_attack["Ast_Score"].astype(int) + df_attack["Gls_Score"].astype(int) + \
                              df_attack["Non-Penalty_goals_Score"].astype(int) + df_attack["Shots_on_Target%_Score"].astype(int) + \
                              df_attack["Goals/Shots_Score"].astype(int) + df_attack["Goal_scored_minus_xG_Score"].astype(int)+ \
                              df_attack["Passes_Leading_to_Shot_Attempt_Score"].astype(int) + df_attack["Dribbles_Leading_to_Goals_Score"].astype(int) + \
                              df_attack["Goal_Creating_Actions_Score"].astype(int) + df_attack["Passes_Leading_to_Goals_Score"].astype(int) + df_attack["Dribbles_Leading_to_Goals_Score"].astype(int)+ \
                              df_attack["Touches_in_Attacking_3rd_Score"].astype(int) + df_attack["Touches_in_Attacking_Penalty_Box_Score"].astype(int)+ \
                              df_attack["Carries_into_Attacking_Penalty_Box_Score"].astype(int) + df_attack["Total_Successful_Dribbles_Score"].astype(int) +df_attack["Total_Failed_Attempts_at_Controlling_Ball_Score"].astype(int)+ \
                              df_attack["%Progressive_Passes_Received_Score"].astype(int) + df_attack["%Completed_passes_that_enter_Penalty_Box_Score"].astype(int) +df_attack["%Aerial_Duel_Total_Score"].astype(int) + \
                              df_attack["%_of_Times_Successfully_Received_Pass_Score"].astype(int)  + df_attack["Tackles_in_Attacking_3rd_Score"].astype(int) + df_attack["Successful_Pressure_%_Score"].astype(int) )
df_attack["Total_Score"].sort_values(ascending = False)




df_attack["Age"].describe().T
df_attack["Total_Score"].quantile(0.90)

df_attack['Performance'] = pd.cut(x=df_attack['Total_Score'], bins=[33,50,72,86,100,113],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])
df_attack["Performance"].value_counts()



df_attack['Age_Cat'] = pd.cut(x=df_attack['Age'], bins=[15, 22, 27, 32, 40],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])
df_attack["Age_Cat"].value_counts()



df_attack["Segment19_20"] = df_attack["Age_Cat"].astype(str) + "_" + df_attack["Performance"].astype(str)




df_attack.loc[(df_attack["Performance"] == "Under_expected"), "Performance_Score_19/20"] = "1"
df_attack.loc[(df_attack["Performance"] == "Open_to_development"), "Performance_Score_19/20"] = "2"
df_attack.loc[(df_attack["Performance"] == "Player_with_high_potential"), "Performance_Score_19/20"] = "3"
df_attack.loc[(df_attack["Performance"] == "High_performance"), "Performance_Score_19/20"] = "4"
df_attack.loc[(df_attack["Performance"] == "Flawless"), "Performance_Score_19/20"] = "5"


###### final_attack
final_attack_19_20 = pd.DataFrame()
final_columns_attack= [col for col in df_attack.columns if col in ["Player", "Segment19_20","Performance_Score_19/20"]]

final_attack_19_20 = df_attack[final_columns_attack]

final_attack_19_20["Segment19_20"].value_counts()

final_attack_19_20[final_attack_19_20["Segment19_20"] == "Young_High_performance"]







##### df leri birleştirme

final_total_df_19_20 = pd.concat([final_defender_19_20, final_midfield_19_20, final_attack_19_20], axis=0)
final_total_df_19_20.to_excel('veriler_2.xlsx', index=False)
final_total_df_19_20.shape


df[df["Club"] == "SD Eibar"]
