#pragma once

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
  #define CLOUDPROC_HELPER_DLL_IMPORT __declspec(dllimport)
  #define CLOUDPROC_HELPER_DLL_EXPORT __declspec(dllexport)
  #define CLOUDPROC_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define CLOUDPROC_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define CLOUDPROC_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define CLOUDPROC_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define CLOUDPROC_HELPER_DLL_IMPORT
    #define CLOUDPROC_HELPER_DLL_EXPORT
    #define CLOUDPROC_HELPER_DLL_LOCAL
  #endif
#endif

// Now we use the generic helper definitions above to define CLOUDPROC_API and CLOUDPROC_LOCAL.
// CLOUDPROC_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// CLOUDPROC_LOCAL is used for non-api symbols.

#define CLOUDPROC_DLL

#ifdef CLOUDPROC_DLL // defined if CLOUDPROC is compiled as a DLL
  #ifdef CLOUDPROC_DLL_EXPORTS // defined if we are building the CLOUDPROC DLL (instead of using it)
    #define CLOUDPROC_API CLOUDPROC_HELPER_DLL_EXPORT
  #else
    #define CLOUDPROC_API CLOUDPROC_HELPER_DLL_IMPORT
  #endif // CLOUDPROC_DLL_EXPORTS
  #define CLOUDPROC_LOCAL CLOUDPROC_HELPER_DLL_LOCAL
#else // CLOUDPROC_DLL is not defined: this means CLOUDPROC is a static lib.
  #define CLOUDPROC_API
  #define CLOUDPROC_LOCAL
#endif // CLOUDPROC_DLL





#define PRINT_AND_THROW(s) do {\
  std::cerr << "\033[1;31mERROR " << s << "\033[0m\n";\
  std::cerr << "at " << __FILE__ << ":" << __LINE__ << std::endl;\
  std::stringstream ss;\
  ss << s;\
  throw std::runtime_error(ss.str());\
} while (0)
#define FAIL_IF_FALSE(expr) if (!(expr)) {\
    PRINT_AND_THROW( "expected true: " #expr);\
  }

#ifdef __CDT_PARSER__
#define BOOST_FOREACH(a,b) for(a;;)
#endif

#define ALWAYS_ASSERT(exp) if (!(exp)) {printf("%s failed in file %s at line %i\n", #exp, __FILE__, __LINE__ ); abort();}
