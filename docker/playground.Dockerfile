FROM ubuntu:latest AS base
RUN apt-get update && apt-get install -y git

WORKDIR /playground

# Clone the repo shallowly, filtering out all blobs initially, and only fetching the gh-pages branch.
# "--filter=blob:none" means blobs (file contents) are only downloaded on demand.
# "--sparse" enables sparse checkout directly from clone.
RUN git clone --depth=1 --branch=gh-pages --filter=blob:none --sparse https://github.com/mitmedialab/prg-raise-playground.git /playground \
 && cd /playground \
 && git sparse-checkout set doodlebot \
 && git fetch origin \
 && git reset --hard origin/gh-pages
# RUN git clone --depth=1 --branch=gh-pages --filter=blob:none --sparse \
#     https://github.com/mitmedialab/prg-raise-playground.git /playground

# # Enable sparse checkout and select only the "doodlebot" directory.
# RUN git sparse-checkout set doodlebot

# After setting sparse-checkout, "git checkout" will update the working copy to include only that folder.
RUN git checkout

CMD git pull && cp -r doodlebot/* /dist