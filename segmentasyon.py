
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

keyword_3 = "(19/20)"
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


defender_columns = [col for col in df.columns if col in ["Position","Age","Player","Nation","MP (20/21)",\
"Min (20/21)", "Goals/Shots (20/21)","Passes Leading to Shot Attempt (20/21)","Defensive Actions Leading to Shot Attempt (20/21)", \
"Touches in Defensive Penalty Box (20/21)", "Touches in Defensive 3rd (20/21)","Touches in Midfield 3rd (20/21)","Touches in Open-play (20/21)",\
"Total Carries (20/21)" ,"Total Distance Carried the Ball (20/21)","% of Times Successfully Received Pass (20/21)" ,"Pass Completion % (All pass-types) (20/21)",\
"Total Tackles Won (20/21)","Total Defensive Blocks (20/21)","Total Shots Blocked (20/21)","Goal Saving Blocks (20/21)", \
"Times blocked a Pass (20/21)", "Aerial Duel Won (20/21)","Aerial Duel Lost (20/21)","Total Loose Balls Recovered (20/21)"]]

df_defender = df_defender[defender_columns]

df_defender.head()

##################
#Total Score Oluşturulması
##################
df_defender["Aerial_Duel_Total"] = df_defender["Aerial Duel Won (20/21)"] - df_defender["Aerial Duel Lost (20/21)"]


df_defender["Age_Score"] = pd.qcut(df_defender["Age"], 5, labels= [5,4,3,2,1])
df_defender["MP_Score"] = pd.qcut(df_defender["MP (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Min_Score"] = pd.qcut(df_defender["Min (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Passes_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_defender["Passes Leading to Shot Attempt (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Defensive_Actions_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_defender["Defensive Actions Leading to Shot Attempt (20/21)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_defender["Touches_in_Defensive_Penalty_Box_Score"] = pd.qcut(df_defender["Touches in Defensive Penalty Box (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Touches_in_Defensive_3rd_Score"] = pd.qcut(df_defender["Touches in Defensive 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Total_Disctance_Score"] = pd.qcut(df_defender["Total Distance Carried the Ball (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Pass_Completion_Score"] = pd.qcut(df_defender["Pass Completion % (All pass-types) (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["%_of_Times_Successfully_Received_Pass_Score"] = pd.qcut(df_defender["% of Times Successfully Received Pass (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["total_Loose_Balls_Recovered_Score"] = pd.qcut(df_defender["Total Loose Balls Recovered (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Total_Tackles_Won_Score"] = pd.qcut(df_defender["Total Tackles Won (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Total_Defensive_Blocks_Score"] = pd.qcut(df_defender["Total Defensive Blocks (20/21)"], 5, labels= [1,2,3,4,5])
df_defender["Aerial_Duel_Total_Score"] = pd.qcut(df_defender["Aerial_Duel_Total"], 5, labels= [1,2,3,4,5])


df_defender["Total_Score"] = (df_defender["MP_Score"].astype(int) +  df_defender["Min_Score"].astype(int) + \
                              df_defender["Passes_Leading_to_Shot_Attempt_Score"].astype(int) + df_defender["Defensive_Actions_Leading_to_Shot_Attempt_Score"].astype(int) + \
                              df_defender["Touches_in_Defensive_Penalty_Box_Score"].astype(int) + df_defender["Touches_in_Defensive_3rd_Score"].astype(int) + \
                              df_defender["%_of_Times_Successfully_Received_Pass_Score"].astype(int) + df_defender["Total_Tackles_Won_Score"].astype(int) + \
                              df_defender["Total_Defensive_Blocks_Score"].astype(int) + df_defender["Aerial_Duel_Total_Score"].astype(int) +\
                              df_defender["Total_Disctance_Score"].astype(int) + df_defender["Pass_Completion_Score"].astype(int) + df_defender["total_Loose_Balls_Recovered_Score"].astype(int))
df_defender["Total_Score"].sort_values(ascending = False)
df_defender.head()
df_defender["Total_Score"].describe().T
df_defender[df_defender["Total_Score"] > 57]

df_defender["Total_Score"].quantile(0.90)

##################
#Segmentleme
##################
df_defender['Performance'] = pd.cut(x=df_defender['Total_Score'], bins=[13, 24, 39, 49, 58, 63],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])


df_defender[df_defender['Performance'] == "High_performance"]
df_defender[df_defender['Age'] < 22]

df_defender["Age"].describe().T

df_defender['Age_Cat'] = pd.cut(x=df_defender['Age'], bins=[17, 22, 27, 32, 37],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])


df_defender["Segment20_21"] = df_defender["Age_Cat"].astype(str) + "_" + df_defender["Performance"].astype(str)




df_defender.loc[(df_defender["Performance"] == "Under_expected"), "Performance_Score_20_21"] = "1"
df_defender.loc[(df_defender["Performance"] == "Open_to_development"), "Performance_Score_20_21"] = "2"
df_defender.loc[(df_defender["Performance"] == "Player_with_high_potential"), "Performance_Score_20_21"] = "3"
df_defender.loc[(df_defender["Performance"] == "High_performance"), "Performance_Score_20_21"] = "4"
df_defender.loc[(df_defender["Performance"] == "Flawless"), "Performance_Score_20_21"] = "5"

df_defender.head()
#final dataframe

final_defender = pd.DataFrame()
final_columns_defender= [col for col in df_defender.columns if col in ["Player", "Segment20_21","Performance_Score_20_21"]]

final_defender = df_defender[final_columns_defender]












df.head()



#######################
#Midfield Segmantation
#######################

df_midfield = pd.DataFrame()
df_midfield =df[df["Position"] == "midfield"]
df_midfield.head()




midfield_columns = [col for col in df.columns if col in ["Player","Age","Position","Total Distance of Completed Progressive Passes (All Pass-types) (20/21)",\
"Ast (20/21)", "Gls (20/21)", "Non-Penalty Goals (20/21)","Min (20/21)","Shots on Target% (20/21)", "Goals/Shots on Target (20/21)","Passes Leading to Shot Attempt (20/21)",\
"Dribbles Leading to Shot Attempt (20/21)","Defensive Actions Leading to Shot Attempt (20/21)" ,"Passes Leading to Goals (20/21)", \
"Touches in Midfield 3rd (20/21)", "Touches in Attacking 3rd (20/21)","Touches in Defensive 3rd (20/21)", "Touches in Open-play (20/21)",\
"Total Distance Carried the Ball (20/21)","Total Distance Carried the Ball in Forward Direction","Number of Times Player was Pass Target (20/21)" ,\
"% of Times Successfully Received Pass (20/21)"  ,"Pass Completion % (All pass-types) (20/21)","Completed passes that enter Final 3rd (20/21)",\
"Total Tackles Won (20/21)","Total Players Tackled + Total Interceptions (20/21)","MP (20/21)","Club","Nation","League"]]

df_midfield = df_midfield[midfield_columns]

df_midfield["Age_Score"] = pd.qcut(df_midfield["Age"], 5, labels= [5,4,3,2,1])
df_midfield["MP_Score"] = pd.qcut(df_midfield["MP (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Min_Score"] = pd.qcut(df_midfield["Min (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Ast_Score"] = pd.qcut(df_midfield["Ast (20/21)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_midfield["Gls_Score"] = pd.qcut(df_midfield["Gls (20/21)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_midfield["Non-Penalty_goals_Score"] = pd.qcut(df_midfield["Non-Penalty Goals (20/21)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_midfield["Goals/Shots_on_Target_Score"] = pd.qcut(df_midfield["Goals/Shots on Target (20/21)"].rank(method = "first"), 5, labels= [1,2,3,4,5])
df_midfield["PassesLeadingtoShotAttempt_Score"] = pd.qcut(df_midfield["Passes Leading to Shot Attempt (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Defensive_Actions_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_midfield["Defensive Actions Leading to Shot Attempt (20/21)"].rank(method = "first"), 5, labels= [1,2,3,4,5])
df_midfield["TouchesinDefensive3rd_score"] = pd.qcut(df_midfield["Touches in Defensive 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Touches in Midfield 3rd_Score"] = pd.qcut(df_midfield["Touches in Midfield 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Touches in Attacking 3rd_Score"] = pd.qcut(df_midfield["Touches in Attacking 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Touches in Open-play_Score"] = pd.qcut(df_midfield["Touches in Open-play (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Number of Times Player was Pass Target_Score"] = pd.qcut(df_midfield["Number of Times Player was Pass Target (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Pass Completion % (All pass-types)_Score"] = pd.qcut(df_midfield["Pass Completion % (All pass-types) (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Completed passes that enter Final 3rd_Score"] = pd.qcut(df_midfield["Completed passes that enter Final 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_midfield["Total Tackles Won_Score"] = pd.qcut(df_midfield["Total Tackles Won (20/21)"], 5, labels= [1,2,3,4,5])


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


df_midfield['Performance'] = pd.cut(x=df_midfield['Total_Score'], bins=[16,32,45,56,68,77],labels=["Under_expected", "Open_to_development", "Player_with_high_potential", "High_performance","Flawless"])

df_midfield['Age_Cat'] = pd.cut(x=df_midfield['Age'], bins=[17, 22, 27, 32, 37],labels=["Young", "Experienced", "Mature", "End_Of_Career" ])

df_midfield["Segment20_21"] = df_midfield["Age_Cat"].astype(str) + "_" + df_midfield["Performance"].astype(str)



df_midfield.loc[(df_midfield["Performance"] == "Under_expected"), "Performance_Score_20_21"] = "1"
df_midfield.loc[(df_midfield["Performance"] == "Open_to_development"), "Performance_Score_20_21"] = "2"
df_midfield.loc[(df_midfield["Performance"] == "Player_with_high_potential"), "Performance_Score_20_21"] = "3"
df_midfield.loc[(df_midfield["Performance"] == "High_performance"), "Performance_Score_20_21"] = "4"
df_midfield.loc[(df_midfield["Performance"] == "Flawless"), "Performance_Score_20_21"] = "5"


df_midfield[df_midfield['Performance'] == "High_performance"].head()

df_midfield.head()


#final dataframe for midfield

final_midfield = pd.DataFrame()
final_columns_midfield= [col for col in df_midfield.columns if col in ["Player", "Segment20_21","Performance_Score_20_21"]]

final_midfield = df_midfield[final_columns_midfield]













#######################
#Attack Segmantation
#######################



df_attack = pd.DataFrame()
df_attack =df[df["Position"] == "attack"]
df_attack.head()
df_attack.shape



attack_columns = [col for col in df.columns if col in ["Age", "MP (20/21)","Player","Club","Position","Nation","League","Min (20/21)", \
                                                   "Gls (20/21)","Ast (20/21)","Non-Penalty Goals (20/21)","Shots on Target% (20/21)",\
                                                   "Goals/Shots (20/21)","Goals Scored minus xG (20/21)","Passes Leading to Shot Attempt (20/21)",\
                                                   "Dribbles Leading to Shot Attempt (20/21)","Goal Creating Actions (20/21)","Passes Leading to Goals (20/21)",\
                                                   "Dribbles Leading to Goals (20/21)","Touches in Attacking 3rd (20/21)","Touches in Attacking Penalty Box (20/21)",\
                                                   "Total Successful Dribbles (20/21)","Carries into Attacking Penalty Box (20/21)","Total Failed Attempts at Controlling Ball (20/21)", \
                                                   "% of Times Successfully Received Pass (20/21)","Progressive Passes Received (20/21)","Completed passes that enter Penalty Box (20/21)",\
                                                   "Aerial Duel Won (20/21)","Aerial Duel Lost (20/21)","Tackles in Attacking 3rd (20/21)","Successful Pressure % (20/21)"]]


df_attack = df_attack[attack_columns]

df_attack.head()


df_attack["Aerial_Duel_Total"] = df_attack["Aerial Duel Won (20/21)"] - df_attack["Aerial Duel Lost (20/21)"]


df_attack["Age_Score"] = pd.qcut(df_attack["Age"], 5, labels= [5,4,3,2,1])
df_attack["MP_Score"] = pd.qcut(df_attack["MP (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Min_Score"] = pd.qcut(df_attack["Min (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Ast_Score"] = pd.qcut(df_attack["Ast (20/21)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_attack["Gls_Score"] = pd.qcut(df_attack["Gls (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Non-Penalty_goals_Score"] = pd.qcut(df_attack["Non-Penalty Goals (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Shots_on_Target%_Score"] = pd.qcut(df_attack["Shots on Target% (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Goals/Shots_Score"] = pd.qcut(df_attack["Goals/Shots (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Goal_scored_minus_xG_Score"] = pd.qcut(df_attack["Goals Scored minus xG (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Passes_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_attack["Passes Leading to Shot Attempt (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Dribbles_Leading_to_Shot_Attempt_Score"] = pd.qcut(df_attack["Dribbles Leading to Shot Attempt (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Goal_Creating_Actions_Score"] = pd.qcut(df_attack["Goal Creating Actions (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Passes_Leading_to_Goals_Score"] = pd.qcut(df_attack["Passes Leading to Goals (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Dribbles_Leading_to_Goals_Score"] = pd.qcut(df_attack["Dribbles Leading to Goals (20/21)"].rank(method="first"), 5, labels= [1,2,3,4,5])
df_attack["Touches_in_Attacking_3rd_Score"] = pd.qcut(df_attack["Touches in Attacking 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Touches_in_Attacking_Penalty_Box_Score"] = pd.qcut(df_attack["Touches in Attacking Penalty Box (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Carries_into_Attacking_Penalty_Box_Score"] = pd.qcut(df_attack["Carries into Attacking Penalty Box (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Total_Successful_Dribbles_Score"] = pd.qcut(df_attack["Total Successful Dribbles (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Total_Failed_Attempts_at_Controlling_Ball_Score"] = pd.qcut(df_attack["Total Failed Attempts at Controlling Ball (20/21)"].rank(method="first"), 5, labels= [5,4,3,2,1])
df_attack["%Progressive_Passes_Received_Score"] = pd.qcut(df_attack["Progressive Passes Received (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["%Completed_passes_that_enter_Penalty_Box_Score"] = pd.qcut(df_attack["Completed passes that enter Penalty Box (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["%Aerial_Duel_Total_Score"] = pd.qcut(df_attack["Aerial_Duel_Total"], 5, labels= [1,2,3,4,5])
df_attack["%_of_Times_Successfully_Received_Pass_Score"] = pd.qcut(df_attack["% of Times Successfully Received Pass (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Tackles_in_Attacking_3rd_Score"] = pd.qcut(df_attack["Tackles in Attacking 3rd (20/21)"], 5, labels= [1,2,3,4,5])
df_attack["Successful_Pressure_%_Score"] = pd.qcut(df_attack["Successful Pressure % (20/21)"], 5, labels= [1,2,3,4,5])




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



df_attack["Segment20_21"] = df_attack["Age_Cat"].astype(str) + "_" + df_attack["Performance"].astype(str)




df_attack.loc[(df_attack["Performance"] == "Under_expected"), "Performance_Score_20_21"] = "1"
df_attack.loc[(df_attack["Performance"] == "Open_to_development"), "Performance_Score_20_21"] = "2"
df_attack.loc[(df_attack["Performance"] == "Player_with_high_potential"), "Performance_Score_20_21"] = "3"
df_attack.loc[(df_attack["Performance"] == "High_performance"), "Performance_Score_20_21"] = "4"
df_attack.loc[(df_attack["Performance"] == "Flawless"), "Performance_Score_20_21"] = "5"


###### final_attack
final_attack = pd.DataFrame()
final_columns_attack= [col for col in df_attack.columns if col in ["Player", "Segment20_21","Performance_Score_20_21"]]

final_attack = df_attack[final_columns_attack]

final_attack["Segment20_21"].value_counts()





final_total_df_20_21[final_total_df_20_21["Segment20_21"] == "Experienced_Flawless"]



#df leri birleştirme


final_total_df_20_21= pd.concat([final_defender, final_midfield, final_attack], axis=0)




################################
#satış fiyatı yorumu
###############################

final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Under_expected"), "Sales_Expectation_Price"] = "Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Open_to_development"), "Sales_Expectation_Price"] = "Low - Mid"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Player_with_high_potential"), "Sales_Expectation_Price"] = "Mid"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_High_performance"), "Sales_Expectation_Price"] = "High"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Flawless"), "Sales_Expectation_Price"] = "Very_High"

final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Under_expected"), "Sales_Expectation_Price"] = "Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Open_to_development"), "Sales_Expectation_Price"] = "Low - Mid"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Player_with_high_potential"), "Sales_Expectation_Price"] = "Mid"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_High_performance"), "Sales_Expectation_Price"] = "Mid - High"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Flawless"), "Sales_Expectation_Price"] = "Very_High"


final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Under_expected"), "Sales_Expectation_Price"] = "Very_Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Open_to_development"), "Sales_Expectation_Price"] = "Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Player_with_high_potential"), "Sales_Expectation_Price"] = "Mid"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_High_performance"), "Sales_Expectation_Price"] = "Mid - High"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Flawless"), "Sales_Expectation_Price"] = "High"


final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Under_expected"), "Sales_Expectation_Price"] = "Very_Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Open_to_development"), "Sales_Expectation_Price"] = "Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Player_with_high_potential"), "Sales_Expectation_Price"] = "Low"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_High_performance"), "Sales_Expectation_Price"] = "Mid"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Flawless"), "Sales_Expectation_Price"] = "Mid - High"






#######Recommend For Action ############
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Under_expected"), "Recommend_For_Action"] = "Considering the potential of these young players, encourage them to improve their performance with extra work and training.\
By taking a long-term approach, support their development and show patience."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Open_to_development"), "Recommend_For_Action"] = "Create special training programs to maximize the potential of these young players.\
Help them gain experience by regularly giving them chances in first team matches."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Player_with_high_potential"), "Recommend_For_Action"] = "Young and high-potential players can be an important part of the team in the future. Focusing on opportunities \
for these players to develop their skills and gain experience can greatly benefit in the long run."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_High_performance"), "Recommend_For_Action"] = "Help these young talents improve their physical condition and technical skills so that they can maintain their high performance.\
Encourage them to show that they are ready to give leadership roles within the team."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Young_Flawless"), "Recommend_For_Action"] = "Help these young talents maximize their physical and technical abilities so that they can maintain their excellent performance.\
Encourage them to evaluate media and marketing opportunities to gain more visibility."



final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Under_expected"), "Recommend_For_Action"] = "Create individual training plans for experienced players to improve their performance and increase their motivation.\
Based on their past accomplishments, allow them to focus more on the team leadership role."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Open_to_development"), "Recommend_For_Action"] = "Take a long-term approach to preparing these experienced players for the future.\
Encourage them to build mentoring relationships with young players and share their knowledge."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Player_with_high_potential"), "Recommend_For_Action"] = "Experienced and high-potential players can make an immediate contribution to the team. Thanks to the experience they have, these players can lead in \
 tough matches and guide young players. At the same time, making a special effort to further develop the potential of these players can increase the team's chances of success"
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_High_performance"), "Recommend_For_Action"] = "Support experienced players to maintain a high level of performance.\
Consider increasing their leadership as one of the key players on the team, giving them more responsibility within the team."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Experienced_Flawless"), "Recommend_For_Action"] = "Support experienced players to maintain flawless performances and strengthen their leadership roles.\
Strategically engage them to enable them to take on more responsibility within the team."



final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Under_expected"), "Recommend_For_Action"] = "Develop a special rehabilitation and training program to help mature players return to their jerseys.\
Set new goals to boost their motivation and rekindle their desire to improve their performance."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Open_to_development"), "Recommend_For_Action"] = "Create individual training and training plans to enable these mature players to develop further.\
Consider giving leadership roles within the team and encourage them to share their experiences with younger players."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Player_with_high_potential"), "Recommend_For_Action"] = "Create a strategic plan to maximize the high potential of these mature players.\
Ensure that they maintain the proper balance of training and rest so that they can maintain their performance."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_High_performance"), "Recommend_For_Action"] = "Help mature players maintain their high level of performance and encourage them to make more impact within the team by increasing their leadership.\
Encourage young players to share their experiences by mentoring them."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "Mature_Flawless"), "Recommend_For_Action"] = "Help mature players maintain flawless performances and encourage them to make more impact within the team by increasing their leadership.\
Encourage young players to share their experiences by mentoring them"




final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Under_expected"), "Recommend_For_Action"] = "Provide specific support and motivation for players nearing the end of their careers to rotate their jerseys.\
Consider mentoring roles or assistant coaching positions to allow the team to benefit from their experience."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Open_to_development"), "Recommend_For_Action"] = "Help players nearing the end of their careers prepare their final stages for the future of the team.\
Support players in thinking about their own post-career plans and training."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Player_with_high_potential"), "Recommend_For_Action"] = "Players who are nearing the end of their careers but still have high potential can be a valuable asset to teams. Thanks to their experience, these players can give advice to young talents and increase \
 the morale of the team with their leadership on the field. The future contributions of these players must be carefully evaluated and aligned with the team's overall strategy."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_High_performance"), "Recommend_For_Action"] = "Provide physical and psychological support to help these players maintain their performance as they approach the end of their careers.\
Take full advantage of their experience by increasing their leadership role within the team."
final_total_df_20_21.loc[(final_total_df_20_21["Segment20_21"] == "End_Of_Career_Flawless"), "Recommend_For_Action"] = "Help players near the end of their careers maintain flawless performances and strengthen their leadership roles.\
Prepare players for mentoring or managerial roles to support their post-career plans."












#final_total_df_çıktı alma

final_total_df_20_21.to_excel('veriler.xlsx', index=False)
final_total_df_20_21.shape

## Sales _ expectation çıktı
Sales_Expectation = pd.DataFrame()
Sales_Expectation_columns= [col for col in final_total_df_20_21.columns if col in ["Player","Sales_Expectation"]]

Sales_Expectation = final_total_df_20_21[Sales_Expectation_columns]

Sales_Expectation.to_excel('Sales_Expectation_.xlsx', index=False)




####### Recommend For Action çıktı ############


Recommend_for_action = pd.DataFrame()
Recommend_for_action_columns= [col for col in final_total_df_20_21.columns if col in ["Player","Recommend_For_Action"]]

Recommend_for_action = final_total_df_20_21[Recommend_for_action_columns]

Recommend_for_action.to_excel('Recommend_for_action.xlsx', index=False)











