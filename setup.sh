mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"fiona16ti@alumni.pcr.ac.id\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml