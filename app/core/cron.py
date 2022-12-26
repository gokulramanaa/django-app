from django_cron import CronJobBase, Schedule
from app.core.modal import NameGenerator


class CronJob(CronJobBase):
    RUN_EVERY_MINS = 1 # every 5 minutes
    RETRY_AFTER_FAILURE_MINS = 1
    schedule = Schedule(run_every_mins=RUN_EVERY_MINS, retry_after_failure_mins=RETRY_AFTER_FAILURE_MINS)
    code = 'cron.my_cron_job'    # a unique code

    def do(self):
        print(NameGenerator().set_values())