import streamlit as st
import pandas as pd
import random
import os

# File to store group assignments
CSV_FILE = "group_assignments.csv"

def load_assignments():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["Name", "Group"])

def save_assignments(df):
    df.to_csv(CSV_FILE, index=False)

def get_available_groups(assignments):
    group_counts = assignments["Group"].value_counts().to_dict()
    available_groups = [i for i in range(1, 11) if group_counts.get(i, 0) < 4]
    return available_groups

def assign_group(name, assignments):
    name = name.strip().title()
    if name in assignments["Name"].values:
        return assignments.loc[assignments["Name"] == name, "Group"].values[0]
    
    available_groups = get_available_groups(assignments)
    if not available_groups:
        return "All groups are full!"
    
    assigned_group = random.choice(available_groups)
    new_entry = pd.DataFrame({"Name": [name], "Group": [assigned_group]})
    assignments = pd.concat([assignments, new_entry], ignore_index=True)
    save_assignments(assignments)
    return assigned_group

# Streamlit UI
st.title("Student Group Assigner")

name = st.text_input("Enter your name:")
if st.button("Get Group Number"):
    if name.strip() and name.replace(" ", "").isalpha():
        assignments = load_assignments()
        group_number = assign_group(name, assignments)
        st.session_state["last_assigned"] = (name, group_number)
        st.success(f"{name}, you have been assigned to Group {group_number}")
    else:
        st.warning("Please enter a valid name (letters only, no symbols or dots).")

if "last_assigned" in st.session_state:
    if st.button("Show My Group"):
        name, group_number = st.session_state["last_assigned"]
        st.info(f"{name}, you are in Group {group_number}")

if st.checkbox("Show all group assignments"):
    assignments = load_assignments()
    st.write(assignments)
