#!/usr/bin/env bash
set +e
root=${1:?evidence root required}
run_dir=${2:-}
mkdir -p "$root/system" "$root/logs" "$root/crash" "$root/jvm" "$root/native"
{
  uname -a
  sw_vers
  system_profiler SPHardwareDataType SPDisplaysDataType
  sysctl -n hw.ncpu
  vm_stat
  df -h
  ps aux
} > "$root/system/host.txt" 2>&1
if [[ -n "${MITHRIL_DYLIB:-}" && -f "$MITHRIL_DYLIB" ]]; then
  file "$MITHRIL_DYLIB" > "$root/native/file.txt" 2>&1
  otool -L "$MITHRIL_DYLIB" > "$root/native/otool-L.txt" 2>&1
  otool -l "$MITHRIL_DYLIB" > "$root/native/otool-l.txt" 2>&1
  nm -gU "$MITHRIL_DYLIB" > "$root/native/nm.txt" 2>&1
  codesign -dvvv "$MITHRIL_DYLIB" > "$root/native/codesign.txt" 2>&1
  shasum -a 256 "$MITHRIL_DYLIB" > "$root/native/sha256.txt" 2>&1
fi
if [[ -n "$run_dir" && -d "$run_dir" ]]; then
  [[ -d "$run_dir/logs" ]] && cp -R "$run_dir/logs/." "$root/logs/"
  [[ -d "$run_dir/crash-reports" ]] && cp -R "$run_dir/crash-reports/." "$root/crash/"
  [[ -d "$run_dir/screenshots" ]] && { mkdir -p "$root/screenshots"; cp -R "$run_dir/screenshots/." "$root/screenshots/"; }
  find "$run_dir" -name 'hs_err_pid*.log' -exec cp {} "$root/jvm/" \; 2>/dev/null
fi
pids=$(pgrep -f 'net.fabricmc.loader.impl.launch.knot.KnotClient|minecraft' || true)
for pid in $pids; do
  jcmd "$pid" VM.flags > "$root/jvm/${pid}-vm-flags.txt" 2>&1 || true
  jcmd "$pid" VM.system_properties > "$root/jvm/${pid}-system-properties.txt" 2>&1 || true
  jcmd "$pid" Thread.print > "$root/jvm/${pid}-threads.txt" 2>&1 || true
  sample "$pid" 3 > "$root/system/${pid}-sample.txt" 2>&1 || true
done
cp -R "$HOME/Library/Logs/DiagnosticReports" "$root/system/DiagnosticReports" 2>/dev/null || true
exit 0
