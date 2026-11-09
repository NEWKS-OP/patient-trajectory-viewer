#!/bin/bash
# Double-click to run the viewer locally.
cd "$(dirname "$0")/web"
[ -d node_modules ] || npm install
(sleep 3 && open http://localhost:5173) &
npm run dev -- --port 5173
