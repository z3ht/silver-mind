const target = "https://localhost:8421";


function onPlayerMove (s, t, piece, newPos, oldPos, orientation) {
  $.post(`${target}/chess/move?Move=${s}${t}`, function(data, status){
    board.position(data);
  });
}


function doComputerMove () {
  $.get(`${target}/chess/next`, function(data, status){
    board.position(data);
  });
}


function undo () {
  $.get(`${target}/chess/undo`, function(data, status){
    board.position(data);
  });
}


function start () {
  $.get(`${target}/chess/start`, function(data, status){
    board.position(data);
  });
}


var config = {
  draggable: true,
  onDrop: onPlayerMove
}

var board = ChessBoard('board', config);

$('#start').on('click', start);
$('#next').on('click', doComputerMove);
$('#undo').on('click', undo);
