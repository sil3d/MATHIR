' run_healthcheck_hidden.vbs
' Launches the MATHIR healthcheck with NO console window at all.
' wscript.exe is a GUI-subsystem binary: it can never create a console,
' so the ~5-min Task Scheduler run cannot flash a window on screen.
' Fully portable: resolves its own directory at runtime (no hardcoded
' username/paths), so the same file ships to every machine.
' Window style 0 = hidden, WaitOnReturn = True keeps wscript alive
' until the healthcheck finishes (so the task shows a real exit code).
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
binDir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1    = binDir & "\auto_start_healthcheck.ps1"
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """ -Quiet", 0, True