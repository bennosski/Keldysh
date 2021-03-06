import time
import atexit

class timed:
    program_start_time = time.time() # assumes profiler is imported at start of code
    functions = {}

    def __init__(self, f):
        self.f = f
        if len(timed.functions)==0:
            atexit.register(self._profile)
        timed.functions[f.__name__] = {'times': []}

    def __get__(self, instance, _):
        self.obj = instance
        return self.__call__

    def __call__(self, *args, **kwargs):
        t0 = time.time()
        if hasattr(self, 'obj'):
            out = self.f(self.obj, *args, **kwargs)
        else:
            out = self.f(*args, **kwargs)
        timed.functions[self.f.__name__]['times'].append(time.time()-t0)
        return out

    def _profile(self):
        program_run_time = time.time() - timed.program_start_time

        fnames = []
        totals = []
        ncalls = []
        total_total = 0

        for fname in timed.functions:
            fnames.append(fname)
            times = timed.functions[fname]['times']
            total = sum(times)
            if len(times)==0:
                tavg = 0
            else:
                tavg = total/len(times)
            total_total += total
            totals.append(total)
            ncalls.append(len(times))
                 
        totals,fnames,ncalls = zip(*sorted(zip(totals,fnames,ncalls), reverse=True))

        percents = [100.0*t/program_run_time for t in totals]
        
        max_name_len    = max(4, max(len(x) for x in fnames))
        max_ncall_len   = max(len(str(n)) for n in ncalls)
        max_percent_len = max(len('{:.1f}'.format(p)) for p in percents)

        space = max_name_len + max_ncall_len + max_percent_len + 36
        s1 = (space-15)//2
        s2 = (space-15)-s1

        print('')
        print(' '+'-'*space+' ')
        print('|'+' '*s1+'Runtime Profile'+' '*s2+'|')
        print('|'+'-'*space+'|')
        print('| {:{}s}   {:{}s}  | {:{}s}  | total time | time/call  |'.format('name', max_name_len, '', max_ncall_len, '  %', max_percent_len))
        print('|'+'-'*space+'|')        
    
        for i,fname in enumerate(fnames):
            if ncalls[i]<=1:
                tavg_str = '          '
            else:
                tavg_str = '{:.3e}s'.format(totals[i]/ncalls[i])

            print('| {:{}s} (x{:{}d}) | {:0{}.1f}% | {:.3e}s | {} |'.format(fname, max_name_len, ncalls[i], max_ncall_len, percents[i], max_percent_len, totals[i], tavg_str))

        print('|'+'-'*space+'|')
        print('| {:{}s}{:{}s}    {:{}s}  | {:.3e}s | {:9s}  |'.format('program', max(7, max_name_len+3), '', max_ncall_len, '', max_percent_len, program_run_time, ''))
        print(' '+'-'*space+' ')
