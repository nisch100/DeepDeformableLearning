var img_img = document.getElementById("img-img");
var txt_img = document.getElementById("txt-img");
var blk_img = document.getElementById("blk-img");
var img_q = document.getElementById("img-q");
var txt_q = document.getElementById("txt-q");
var blk_q = document.getElementById("blk-q");
var img_choices_wrapper = document.getElementById("img-choices-wrapper");
var img_json = document.getElementById("img-json");
var txt_choices_wrapper = document.getElementById("txt-choices-wrapper");
var txt_json = document.getElementById("txt-json");
var blk_json = document.getElementById("blk-json");

// updates a question
// type = "img" / "txt" / "blk"
// id = index in the CHOOSE_IMAGE_LS / CHOOSE_TEXT_LS / FILL_IN_BLANK arrays
// CHOOSE_IMAGE_LS / CHOOSE_TEXT_LS / FILL_IN_BLANK arrays are defined in examples-meta.js
// They include all the data.json for the examples
function update_q(type, id) {
    if (type === "img") {
        // update img
        img_img.src = CHOOSE_IMAGE_LS[id].path + "/image.png";
        // update question
        img_q.innerHTML = "<strong>Question: </strong>" + CHOOSE_IMAGE_LS[id].question;
        // create the elements for the image choices 
        img_choices_wrapper.innerHTML = create_img_choices(id);
        // take out the info we want from the jsons and display them in the correct place
        var displayed_data = (({question, choices, answer, ques_type, grade, label}) => ({question, choices, answer, ques_type, grade, label}))(CHOOSE_IMAGE_LS[id]);
        img_json.innerText = JSON.stringify(displayed_data, null, 4);
    } else if (type === "txt") {
        txt_img.src = CHOOSE_TEXT_LS[id].path + "/image.png";
        txt_q.innerHTML = "<strong>Question: </strong>" + CHOOSE_TEXT_LS[id].question;
        txt_choices_wrapper.innerHTML = create_txt_choices(id);
        var displayed_data = (({question, choices, answer, ques_type, grade, label}) => ({question, choices, answer, ques_type, grade, label}))(CHOOSE_TEXT_LS[id]);
        txt_json.innerText = JSON.stringify(displayed_data, null, 4);
    } else {
        blk_img.src = FILL_IN_BLANK_LS[id].path + "/image.png";
        blk_q.innerHTML = "<strong>Question: </strong>" + FILL_IN_BLANK_LS[id].question;
        var displayed_data = (({question, choices, answer, ques_type, grade, label}) => ({question, choices, answer, ques_type, grade, label}))(FILL_IN_BLANK_LS[id]);
        blk_json.innerText = JSON.stringify(displayed_data, null, 4);
    }
}

function create_img_choices(id) {
    var str = "";
    for (let i = 0; i < CHOOSE_IMAGE_LS[id].choices.length; i++) {
        path = CHOOSE_IMAGE_LS[id].path + "/" + CHOOSE_IMAGE_LS[id].choices[i];
        str += `<img class="img-choice" src="${path}">`;
    }
    return str;
}

function create_txt_choices(id) {
    var str = "";
    for (let i = 0; i < CHOOSE_TEXT_LS[id].choices.length; i++) {
        str += `<div class="txt-choice"> ${CHOOSE_TEXT_LS[id].choices[i]} </div>`;
    }
    return str;
}

// current question on display
var img_id, txt_id, blk_id;
img_id = txt_id = blk_id = 0;
update_q("img", img_id);
update_q("txt", txt_id);
update_q("blk", blk_id);

// prev/next button functionalities
var img_next = document.getElementById("img-next");
var img_prev = document.getElementById("img-prev");
img_next.addEventListener("click", () => {
    img_id++;
    if (img_id >= CHOOSE_IMAGE_LS.length)
        img_id = 0;
    update_q("img", img_id);
});
img_prev.addEventListener("click", () => {
    img_id--;
    if (img_id < 0)
        img_id = CHOOSE_IMAGE_LS.length-1;
    update_q("img", img_id);
});

var txt_next = document.getElementById("txt-next");
var txt_prev = document.getElementById("txt-prev");
txt_next.addEventListener("click", () => {
    txt_id++;
    if (txt_id >= CHOOSE_TEXT_LS.length)
        txt_id = 0;
    update_q("txt", txt_id);
});
txt_prev.addEventListener("click", () => {
    txt_id--;
    if (txt_id <0)
        txt_id = CHOOSE_TEXT_LS.length - 1;
    update_q("txt", txt_id);
});

var blk_next = document.getElementById("blk-next");
var blk_prev = document.getElementById("blk-prev");
blk_next.addEventListener("click", () => {
    blk_id++;
    if (blk_id >= FILL_IN_BLANK_LS.length)
        blk_id = 0;
    update_q("blk", blk_id);
});
blk_prev.addEventListener("click", () => {
    blk_id--;
    if (blk_id < 0)
        blk_id =  FILL_IN_BLANK_LS.length - 1;
    update_q("blk", blk_id);
});



