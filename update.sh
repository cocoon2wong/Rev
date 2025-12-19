###
 # @Author: Conghao Wong
 # @Date: 2025-03-24 17:16:55
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2025-12-19 10:31:07
 # @Github: https://cocoon2wong.github.io
 # Copyright 2025 Conghao Wong, All Rights Reserved.
###

rm -r ./_data
rm -r ./_includes
rm -r ./_layouts
rm -r ./assets

bash preprocess.sh

bundle exec jekyll serve