function predict(){
    var total_sqft = document.form.total_sqft.value
    var location = document.form.location.value
    var bhk = document.form.bhk.value
    var num_bathrooms = document.form.bath.value

    if(total_sqft == "" || location == "" || bhk == "" || num_bathrooms == ""){
        alert("You have to enter all data - area in sqft, location, bhk and num_bathrooms!")
        return false;
    }
    else{
        alert('You are searching price for house: ' + total_sqft + ', ' + location + ', ' + bhk + ', ' + num_bathrooms);
        let niz = []
        if(localStorage.getItem("searches")!=null){
           niz = JSON.parse(localStorage.getItem("searches"))
        }

        let obj = {
           total_sqft: total_sqft,
           location: location,
           bhk: bhk,
           num_bathrooms: num_bathrooms
        }

        niz.push(obj)
        localStorage.setItem("searches", JSON.stringify(niz))
        alert("Thanks for using this app!")
        return true
    }
}
