
export http_proxy=http://127.0.0.1:1087
export https_proxy=http://127.0.0.1:1087
cd /Users/kichikudetokumeikun/Desktop/PhDandMEng/CSU/Web/Blog
git add .
git commit -m "add info"
git push -u superman master
osascript -e 'tell application"Terminal" to close (every window whose name contains".command")' &
exit
