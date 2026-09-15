import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


# -----------------------------------------
# 1. Create the LLM
# -----------------------------------------

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"]
)


# -----------------------------------------
# 2. Create Role + Few-Shot Prompt
# -----------------------------------------

messages = [

    # ROLE
    SystemMessage(
        content="""
You are a friendly Nutrition Assistant.

Your job is to help users understand whether
a food choice is generally healthy.

Classify food into one of these categories:

Healthy
Moderate
Avoid

Always respond using this format:

Category: <category>

Reason:
<short explanation>

Suggestion:
<short suggestion>

Keep your answer simple and suitable for beginners.
"""
    ),


    # -----------------------------------------
    # FEW-SHOT EXAMPLE 1
    # -----------------------------------------

    HumanMessage(
        content="I want to eat an apple."
    ),

    AIMessage(
        content="""
Category: Healthy

Reason:
Apple contains fiber, vitamins and natural nutrients.

Suggestion:
Apple can be included as part of a balanced diet.
"""
    ),


    # -----------------------------------------
    # FEW-SHOT EXAMPLE 2
    # -----------------------------------------

    HumanMessage(
        content="I want to eat pizza."
    ),

    AIMessage(
        content="""
Category: Moderate

Reason:
Pizza can contain a lot of refined carbohydrates,
cheese, salt and calories depending on the preparation.

Suggestion:
Eat it occasionally and prefer vegetables and
moderate cheese as toppings.
"""
    ),


    # -----------------------------------------
    # FEW-SHOT EXAMPLE 3
    # -----------------------------------------

    HumanMessage(
        content="I want to drink sugary soda every day."
    ),

    AIMessage(
        content="""
Category: Avoid

Reason:
Sugary drinks can contain a large amount of
added sugar and provide little nutritional value.

Suggestion:
Prefer water or unsweetened drinks for regular use.
"""
    )
]


# -----------------------------------------
# 3. Streamlit UI
# -----------------------------------------

st.title("🥗 Your Personal Healthy Food Assistant")

st.write(
    "Ask me whether a food or drink is generally "
    "healthy, moderate, or best avoided."
)


food = st.chat_input(
    "What food or drink are you thinking about?"
)


# -----------------------------------------
# 4. Process User Question
# -----------------------------------------

if food:

    st.chat_message("user").write(food)

    # Create a copy so the examples don't
    # permanently grow on every Streamlit rerun
    conversation = messages.copy()

    conversation.append(
        HumanMessage(
            content=f"I want to eat or drink {food}."
        )
    )


    # -----------------------------------------
    # 5. Call Groq
    # -----------------------------------------

    with st.spinner("Thinking..."):

        try:
            response = model.invoke(conversation)

            st.chat_message("assistant").write(
                response.content
            )

        except Exception as e:

            st.error(
                "Something went wrong while contacting Groq."
            )

            st.exception(e)