#!/bin/sh
# docker-entrypoint.sh


if [[ -f ./modaldata.pkl ]]
then
  touch ./modaldata.pkl
fi

# If this is going to be a cron container, set up the crontab.
# if [ "$1" = cron ]; then
#   ./manage.py runcrons
# fi

./manage.py runcrons
# Launch the main container command passed as arguments.
exec "$@"