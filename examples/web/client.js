const target = "https://localhost:8421";


function onPlayerMove (source, t, piece, newPos, oldPos, orientation) {
  const fen_result = ChessBoard.objToFen(newPos);
  $.post(`${target}/chess/move?New_Position=${fen_result}`, function(data, status){
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
