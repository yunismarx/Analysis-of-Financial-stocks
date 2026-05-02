import streamlit as st

if 'test_dict' not in st.session_state:
    st.session_state.test_dict = {'a': 1}

st.write(st.session_state.test_dict)

with st.form('form'):
    name = st.text_input('Key')
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.session_state.test_dict[name] = 2
        st.write('Saved!')
        st.rerun()
