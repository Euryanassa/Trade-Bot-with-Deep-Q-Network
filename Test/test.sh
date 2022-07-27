
printf "▀▀█▀▀ ▒█▀▀█ ░█▀▀█ ▒█▀▀▄ ▒█▀▀▀ ▒█▀▀█ ▒█▀▀▀█ ▀▀█▀▀ ░░ ▀█░█▀ ░ ▄█░ ░ █▀▀█\n░▒█░░ ▒█▄▄▀ ▒█▄▄█ ▒█░▒█ ▒█▀▀▀ ▒█▀▀▄ ▒█░░▒█ ░▒█░$

printf "\n\n\nMIT License\n\nCopyright (c) 2022 Altemur Çelikayar"
printf "\n\n\n[\033[0;36mInfo\033[0m]Training Begins\n\n\n"

python trader_file.py --stocks AAPL MSI SBUX \
                      --start_date '2021-01-01' \
                      --end_date '2022-07-23'
                      --initial_investment 20000 \
                      --gamma 0.95 \
                      --epsilon 1.0 \
                      --epsilon_min 0.01 \
                      --epsilon_decay 0.995 \
                      --num_episodes 50 \
                      --save_model True

