#!/usr/bin/env sh

apply=false
quiet=false
style="LLVM"

for f in $(git ls-files -- '*.h' '*.hpp' '*.c' '*.cc' '*.cpp' | grep -v '/onnx_to_linalg'); do

  if [ $quiet == false ];
  then
    echo "Analysing: [$f]"
  fi

  filename="$(basename -- $f)"
  pathname="$(dirname -- $f)"

  ##
  ## clang-format
  ##

  pushd $pathname > /dev/null
  cdiff=`git diff --no-index --color -- "$filename" <(clang-format --style=$style "$filename")`
  if [[ -n "$cdiff" && "$apply" == true ]]; then
    udiff=`git diff --no-index --no-color -- "$filename" <(clang-format --style=$style "$filename")`
    echo "udiff" | patch -p1
  fi
  popd > /dev/null

  ##
  ## cpplint
  ##

  flt="-whitespace/indent,"
  flt+="-whitespace/comments,"
  flt+="+readability/missing-final-newline"
  clint=`cpplint --quiet --filter=${flt} "$f" 2>&1`

  ##
  ## display errors
  ##

  if [[ -n "$cdiff" || -n "$clint" ]]; then
    echo
    echo -e "------->>>--[\e[32m$f\e[0m]-->>>---------"
    echo "$cdiff" | tail -n +5
    echo "$clint"
    echo -e "-------<<<--[\e[32m$f\e[0m]--<<<---------"
    echo
  fi

done
