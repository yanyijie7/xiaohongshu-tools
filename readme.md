python3 -m venv .env

.env
.DS_Store

git touch out/.gitkeep

git push -u origin main

# 推送本地dev分支到远端，并将本地dev与远端dev关联（-u表示设置上游分支）
git push -u origin dev

//开启虚拟环境
source .env/bin/activate



pip install numpy
pip install pillow