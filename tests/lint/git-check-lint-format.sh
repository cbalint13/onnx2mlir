#!/usr/bin/env sh

apply=false
quiet=false
style="LLVM"

while [ "$1" != "" ]; do
  case "$1" in
    --apply | -a)
      apply=true
      ;;
    --quiet | -q)
      quiet=true
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

if ! command -v clang-format &> /dev/null; then
  echo "ERROR: clang-format is not installed."
  exit 1
fi

if ! command -v cpplint &> /dev/null; then
  echo "ERROR: cpplint is not installed."
  exit 1
fi

if ! command -v black &> /dev/null; then
  echo "ERROR: black is not installed."
  exit 1
fi

if ! command -v pylint &> /dev/null; then
  echo "ERROR: pylint is not installed."
  exit 1
fi

for f in $(git ls-files -- '*.h' '*.hpp' '*.c' '*.cc' '*.cpp'); do

  if [ $quiet == false ];
  then
    echo "Analysing: [$f]"
  fi

  ##
  ## clang-format
  ##

  cdiff=`git diff --no-index --color -- $f <(clang-format --style=$style $f)`
  if [[ -n "$cdiff" && $apply == true ]]; then
    clang-format -i --style=$style $f
    echo "Formatted: [$f]"
  fi

  ##
  ## cpplint
  ##

  flt="-whitespace/indent,"
  flt+="-whitespace/comments,"
  flt+="+readability/missing-final-newline"
  clint=`cpplint --quiet --filter=${flt} $f 2>&1`

  # display errors
  if [[ -n "$cdiff" || -n "$clint" ]]; then
    echo
    echo -e "------->>>--[\e[32m$f\e[0m]-->>>---------"
    echo "$cdiff" | tail -n +5
    echo "$clint"
    echo -e "-------<<<--[\e[32m$f\e[0m]--<<<---------"
    echo
  fi

done

if ! which black &> /dev/null; then
  echo "ERROR: black is installed."
  exit 1
fi

for f in $(git ls-files -- '*.py'); do

  if [ $quiet == false ];
  then
    echo "Analysing: [$f]"
  fi

  ##
  ## black
  ##

  bdiff=`black --quiet --check --color --diff $f`
  if [[ -n "$bdiff" && $apply == true ]]; then
    black --quiet $f
    echo "Formatted: [$f]"
  fi


  ##
  ## pylint
  ##

  pdiff=`pylint --score=no --disable=no-name-in-module,import-error $f 2>&1`

  # display errors
  if [[ -n "$bdiff" || -n "$pdiff" ]]; then
    echo
    echo -e "------->>>--[\e[32m$f\e[0m]-->>>---------"
    echo "$bdiff" | tail -n +2
    echo "$pdiff"
    echo -e "-------<<<--[\e[32m$f\e[0m]--<<<---------"
    echo
  fi

done

exit 0
