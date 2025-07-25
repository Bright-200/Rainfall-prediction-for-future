import streamlit as st
import pandas as pd
import random
import os
import nltk
from nltk.corpus import words

nltk.download('words')
word_list = set(words.words())

# File to store group assignments
CSV_FILE = "group_assignments.csv"
TOPICS_FILE = "group_topics.csv"
LEADERS_FILE = "group_leaders.csv"

# Load and save functions
def load_assignments():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["Name", "Group"])

def save_assignments(df):
    df.to_csv(CSV_FILE, index=False)

def load_topics():
    if os.path.exists(TOPICS_FILE):
        return pd.read_csv(TOPICS_FILE)
    return pd.DataFrame(columns=["Group", "Topic"])

def save_topics(df):
    df.to_csv(TOPICS_FILE, index=False)

def load_leaders():
    if os.path.exists(LEADERS_FILE):
        return pd.read_csv(LEADERS_FILE)
    return pd.DataFrame(columns=["Group", "Leader"])

def save_leaders(df):
    df.to_csv(LEADERS_FILE, index=False)

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

def assign_topics():
    topics = ["Topic A", "Topic B", "Topic C", "Topic D", "Topic E", "Topic F", "Topic G", "Topic H", "Topic I", "Topic J"]
    random.shuffle(topics)
    topic_df = pd.DataFrame({"Group": list(range(1, 11)), "Topic": topics})
    save_topics(topic_df)
    return topic_df

def assign_leader(name, group):
    leaders = load_leaders()
    if group in leaders["Group"].values:
        return "A leader is already assigned to this group."
    new_leader = pd.DataFrame({"Group": [group], "Leader": [name]})
    leaders = pd.concat([leaders, new_leader], ignore_index=True)
    save_leaders(leaders)
    return "Leader assigned successfully."

# Streamlit UI
st.title("Student Group Assigner")

name = st.text_input("Enter your name:")

if st.button("Validate Name"):
    if name.strip() and name.replace(" ", "").isalpha():
        if any(word.lower() in word_list for word in name.split()):
            st.success("Valid name.")
        else:
            if st.button("Approve Name Manually"):
                st.session_state["approved_name"] = name.strip().title()
                st.success("Name manually approved.")
            else:
                st.warning("The name does not look valid. Approve manually if necessary.")
    else:
        st.warning("Please enter a valid name (letters only, no symbols or dots).")

if "approved_name" in st.session_state and st.button("Get Group Number"):
    assignments = load_assignments()
    group_number = assign_group(st.session_state["approved_name"], assignments)
    st.session_state["last_assigned"] = (st.session_state["approved_name"], group_number)
    st.success(f"{st.session_state['approved_name']}, you have been assigned to Group {group_number}")

if "last_assigned" in st.session_state:
    if st.button("Show My Group"):
        name, group_number = st.session_state["last_assigned"]
        st.info(f"{name}, you are in Group {group_number}")

if st.checkbox("Show all group assignments"):
    assignments = load_assignments()
    st.write(assignments)

st.title("Group Leader Selection")
if "last_assigned" in st.session_state:
    name, group_number = st.session_state["last_assigned"]
    if st.button("Become Group Leader"):
        result = assign_leader(name, group_number)
        st.success(result)

if st.checkbox("Show Group Leaders"):
    leaders = load_leaders()
    st.write(leaders)

st.title("Group Topic Assigner")
if st.button("Assign Topics to Groups"):
    topic_df = assign_topics()
    st.success("Topics have been assigned to groups.")
    st.write(topic_df)

leader_name = st.text_input("Enter group leader name to get topic:")
if st.button("Get Topic"):
    leaders = load_leaders()
    if leader_name.strip().title() in leaders["Leader"].values:
        group = leaders.loc[leaders["Leader"] == leader_name.strip().title(), "Group"].values[0]
        topics = load_topics()
        topic = topics.loc[topics["Group"] == group, "Topic"].values[0]
        st.success(f"Group {group}, led by {leader_name.strip().title()}, has been assigned: {topic}")
    else:
        st.warning("This person is not a group leader.")

if st.checkbox("Show Group Topics"):
    topics = load_topics()
    st.write(topics)
