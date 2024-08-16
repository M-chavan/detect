<?php
$username = 'vyassa44_gkmes';
$password = 'Washi@#1234';   
$database = 'vyassa44_gkm';         
$host = '192.250.231.31';                        
$mysqli = new mysqli($host,$username, $password, $database);

if ($mysqli->connect_error) {
    die('Connect Error (' . $mysqli->connect_errno . ') '
            . $mysqli->connect_error);
    mysqli_close();
    echo "error";
   }

   
?>
