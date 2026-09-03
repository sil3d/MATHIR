' mathir_daemon_hidden.vbs
' Launches the MATHIR auto-start (daemon + proxy) with NO console window at
' all (logon task). wscript.exe is a GUI-subsystem binary: it can never
' create a console, so nothing flashes at logon.
' Delegates to auto_start.bat (the same launcher the healthcheck uses), which
' resolves the Python interpreter dynamically (PATH -> py launcher -> common
' install locations) and is idempotent (netstat port checks -- concurrent
' runs cannot spawn duplicate daemons/proxies).
' Fully portable: the .bat path resolves relative to this file's directory,
' so the same file ships to every machine and every user.
' Window style 0 = hidden; WaitOnReturn = False (auto_start.bat returns
' quickly; the daemon and proxy keep running in the background).
Set fso = CreateObject("Scripting.FileSystemObject")
Dim shell : Set shell = CreateObject("WScript.Shell")
Dim BAT_PATH : BAT_PATH = fso.GetParentFolderName(WScript.ScriptFullName) & "\auto_start.bat"
shell.Run """" & BAT_PATH & """", 0, False
Set shell = Nothing