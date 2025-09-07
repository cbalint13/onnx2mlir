#!/usr/bin/env sh

apply=false
quiet=false
style="LLVM"
filepath=""

for f in $(git ls-files -- '*.h' '*.hpp' '*.c' '*.cc' '*.cpp' | grep -v '/onnx_to_linalg'); do

  echo
  echo "-----------------------////---------------------"
  echo "Analysing: [$f]"
  filename="$(basename -- $f)"
  pathname="$(dirname -- $f)"
  pushd $pathname > /dev/null
  clang-format --style=$style $filename > /tmp/$filename.tmp
  if [ $quiet == false ]
  then
    unbuffer git --no-pager diff $filename /tmp/$filename.tmp | tail -n +5
  fi
  git --no-pager diff $filename /tmp/$filename.tmp > /tmp/$filename.dif
  if [ $apply == true ]
  then
    patch -p1 < /tmp/$filename.dif
  fi
  rm -rf /tmp/$filename.???
  popd > /dev/null
  cpplint --filter=-whitespace/comments,-whitespace/indent $f

done
