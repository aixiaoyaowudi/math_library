/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_TOOLS_TOOLS_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_TOOLS_TOOLS_H_

#include <chrono>

namespace tools
{
class timer
{
private:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::duration<double, std::ratio<1> > second;
  std::chrono::time_point<clock> start_time;
  double accumulated_time;
  bool running;

public:
  timer ();
  void start ();
  double stop ();
  double accumulated ();
  double lap ();
  void reset ();
  bool get_state ();
};
class progress_bar
{
private:
  uint32_t total_work;
  uint32_t next_update;
  uint32_t call_diff;
  uint32_t work_done;
  uint16_t old_percent;
  timer _timer;
  void clear_console_line () const;

public:
  void start (uint32_t total_work);
  void update (uint32_t work_done0, bool is_dynamic = true);
  progress_bar &operator++ ();
  double stop ();
  double time_it_took ();
  uint32_t cells_processed () const;
  ~progress_bar ();
};
}

#endif