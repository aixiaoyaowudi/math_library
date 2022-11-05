/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <tools/tools.h>

#if defined(_OPENMP)
#include <omp.h>
#else
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

namespace tools
{
/*
	* Code from https://stackoverflow.com/questions/28050669/can-i-report-progress-for-openmp-tasks
	*/
timer::timer ()
{
  accumulated_time = 0;
  running          = false;
}
void
timer::start ()
{
  if (running)
    throw std::runtime_error ("Timer was already started!");
  running    = true;
  start_time = clock::now ();
}
double
timer::stop ()
{
  if (!running)
    throw std::runtime_error ("Timer was already stopped!");
  accumulated_time += lap ();
  running = false;

  return accumulated_time;
}
double
timer::accumulated ()
{
  if (running)
    throw std::runtime_error ("Timer is still running!");
  return accumulated_time;
}
double
timer::lap ()
{
  if (!running)
    throw std::runtime_error ("Timer was not started!");
  return std::chrono::duration_cast<second> (clock::now () - start_time)
      .count ();
}
void
timer::reset ()
{
  accumulated_time = 0;
  running          = false;
}
bool
timer::get_state ()
{
  return running;
}
void
progress_bar::clear_console_line () const
{
  std::cerr << "\r\033[2K" << std::flush;
}
void
progress_bar::start (uint32_t total_work)
{
  _timer = timer ();
  _timer.start ();
  this->total_work = total_work;
  next_update      = 0;
  call_diff        = total_work / 200;
  old_percent      = 0;
  work_done        = 0;
  clear_console_line ();
}
void
progress_bar::update (uint32_t work_done0, bool is_dynamic)
{
  if (omp_get_thread_num () != 0)
    return;
  work_done = work_done0;
  if (work_done < next_update)
    return;
  next_update += call_diff;
  uint16_t percent;
#ifdef __INTEL_COMPILER
  percent = (uint8_t) ((uint64_t)work_done * omp_get_num_threads () * 100
                       / total_work);
#else
  if (is_dynamic)
    percent = (uint8_t) ((uint64_t)work_done * 100 / total_work);
  else
    percent = (uint8_t) ((uint64_t)work_done * omp_get_num_threads () * 100
                         / total_work);
#endif
  if (percent > 100)
    percent = 100;
  if (percent == old_percent)
    return;
  old_percent = percent;
  std::cerr << "\r\033[2K[" << std::string (percent / 2, '=')
            << std::string (50 - percent / 2, ' ') << "] (" << percent
            << "% - " << std::fixed << std::setprecision (1)
            << _timer.lap () / percent * (100 - percent) << "s - "
            << omp_get_num_threads () << " threads)" << std::flush;
}
progress_bar &
progress_bar::operator++ ()
{
  if (omp_get_thread_num () != 0)
    return *this;
  work_done++;
  update (work_done);
  return *this;
}
double
progress_bar::stop ()
{
  clear_console_line ();
  _timer.stop ();
  return _timer.accumulated ();
}
double
progress_bar::time_it_took ()
{
  return _timer.accumulated ();
}
uint32_t
progress_bar::cells_processed () const
{
  return work_done;
}
progress_bar::~progress_bar ()
{
  if (_timer.get_state ())
    this->stop ();
}
}