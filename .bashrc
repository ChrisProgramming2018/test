# Eternal bash history.
# ---------------------
# Undocumented feature which sets the size to "unlimited".
# http://stackoverflow.com/questions/9457233/unlimited-bash-history
export HISTFILESIZE=
export HISTSIZE=
export HISTTIMEFORMAT="[%F %T] "
# Change the file location because certain bash sessions truncate .bash_history file upon close.
# http://superuser.com/questions/575479/bash-history-truncated-to-500-lines-on-each-login
export HISTFILE=~/.bash_eternal_history
# Force prompt to write history after every command.
# http://superuser.com/questions/20900/bash-history-loss
PROMPT_COMMAND="history -a; $PROMPT_COMMAND"


alias startsim="/export/leiningc/UnrealEngine/Engine/Binaries/Linux/UE4Editor /export/leiningc/ue_world_2018/audicup_world_2/audicup_world.uproject"
alias robot="source /export/leiningc/miniconda3/bin/activate /export/leiningc/miniconda3/envs/surreal"
alias kuka="source /export/leiningc/miniconda3/bin/activate kuka"
alias tf="source /export/leiningc/miniconda3/bin/activate tf"
alias tb="tensorboard --logdir=runs"
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/leiningc/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/leiningc/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

PS1="\u@\h \[\033[32m\]\w - \$(parse_git_branch)\[\033[00m\] $ "

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/export/leiningc/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/export/leiningc/miniconda3/etc/profile.d/conda.sh" ]; then
#        . "/export/leiningc/miniconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/export/leiningc/miniconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<

